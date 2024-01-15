#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__device__ __managed__ u32 gtime = 0;

// Init VCB
__device__ void init_VCB(FileSystem *fs) {
  for (int i=0;i<fs->SUPERBLOCK_SIZE;i++){
    fs->VCB[i] = 0;
  }
}

// Deep copy the file name
__device__ char* deep_copy(char *name, char *target){
  char* tmp = name;
  while(*name++ = *target++){
    ;
  }
  return tmp;
}

// Compare the file name
__device__ bool name_compare(char *name, char *target){
  while(*target){
    if(*name++!=*target++) return false;
  }
  if(*name) return false;
  return true;
}

// Search the file name in the FCB table
__device__ int name_search(FileSystem *fs, char *s, int type, int dir){
  for(int i=0;i<fs->FCB_next;i++){
    FCB_entry fe = fs->FCB_table[i];
    if(dir==fe.parent && name_compare(fe.file_name,s)){
      fs->target_FCB_entry = i;
      if(type==0) return fe.start;
      return i;
    }
  }
  return -1;
}

// data move in the Content when the number of file block changes
__device__ void data_move(FileSystem *fs, int start, int end){
  // Move up.
  if(start>end){
    for(int i=0;i<fs->Content_next-start;i++){
      fs->Content[end+i] = fs->Content[start+i];
    }
    for(int i=fs->Content_next-1;i>=fs->Content_next-(start-end);i--){
      fs->Content[i] = NULL;
    }
  }
  // Move down.
  else{
    for(int i=fs->Content_next+end-start-1;i>=end;i--){
      fs->Content[i] = fs->Content[i-(end-start)];
    }
  }
  fs->Content_next-=start-end;
}

// Update FCB Table when the number of file block changes
__device__ void FCB_update(FileSystem *fs, int start, int end){
  for (int i=fs->target_FCB_entry+1;i<fs->FCB_next;i++){
    if(fs->FCB_table[i].start!=-1){
      fs->FCB_table[i].start+=end-start;
    }
  }
}

// Get the str length
__device__ int my_strlen(const char* file_name)
{
	int count = 0;
	while (*file_name != '\0')
	{
		count++;
		file_name++;
	}
	return count;
}

// Cat two string
__device__ char* my_strcat(char* end, const char* source)
{
	char* tem = end;
	while (*end != '\0')
	{
		end++;
	}
    while (*end++ = *source++)
	{
        ;
    }
    return tem;
}

// Find the largest modified time
__device__ int find_largest_time(FileSystem *fs, int prev_largest){
  int largest = -1;
  int target = -1;
  for(int i=0;i<fs->FCB_next;i++){
    int time = fs->FCB_table[i].time;
    if (fs->FCB_table[i].parent == fs->current_dir && time < prev_largest && time > largest){
        largest = time;
        target = i;
    }
  }
  printf("%s ",fs->FCB_table[target].file_name);
  if(fs->FCB_table[target].file_type == 1) printf("d\n");
  if(fs->FCB_table[target].file_type == 0) printf("\n"); 
  return largest;
}

// Find the largest file size
__device__ int find_largest_size(FileSystem *fs, int prev_largest, int create_order){
  int largest = -1;
  int target = -1;
  for(int i=0;i<fs->FCB_next;i++){
    int size = fs->FCB_table[i].size;
    if (fs->FCB_table[i].parent==fs->current_dir && size == prev_largest && i>create_order){
      target = i;
      break;
    }else if(fs->FCB_table[i].parent==fs->current_dir && size<prev_largest && size>largest){
      largest=size;
      target = i;
    }
  }
  printf("%s %d ",fs->FCB_table[target].file_name,fs->FCB_table[target].size);
  if(fs->FCB_table[target].file_type == 1) printf("d\n");
  if(fs->FCB_table[target].file_type == 0) printf("\n"); 
  return target;
}

// Count the number of file under specific dir
__device__ int count_file_under_dir(FileSystem *fs){
  int count = 0;
  for(int i=0;i<fs->FCB_next;i++){
    if(fs->FCB_table[i].parent==fs->current_dir) count++;
  }
  return count;
}

// Update the parent FCB entry when remove FCB entry
__device__ void update_parent(FileSystem *fs, int parent){
  for (int i=parent+1;i<fs->FCB_next;i++){
    if(fs->FCB_table[i].parent==parent){
      fs->FCB_table[i].parent--;
    }
  }
}

// Judge whether the file is belong to the specific dir
__device__ int belong_to_dir(FileSystem *fs, int target, int dir){
  int parent = fs->FCB_table[target].parent;
  int type = fs->FCB_table[target].file_type;
  while(parent!=-1){
    if(parent==dir && type==0) return 0;
    if(parent==dir && type==1) return 1;
    parent = fs->FCB_table[parent].parent;
  }
  return -1;
}

// Get all the dir and file which are belong to specific dir
__device__ void get_all_dir_and_file(FileSystem *fs, int file){
  for(int i=file;i<fs->FCB_next;i++){
    int type = belong_to_dir(fs,i,file);
    if(type==0){
      fs->remove_file[fs->remove_file_count++] = i;
    }else if(type==1){
      fs->remove_dir[fs->remove_dir_count++] = i;
    }
  }  
}

// Remove specific dir
__device__ void remove_dir(FileSystem *fs, int file_index){
  FCB_entry fe = fs->FCB_table[file_index];
  // Update parent
  if(fe.parent!=-1){
    fs->FCB_table[fe.parent].size -= (my_strlen(fe.file_name)+1);  
  }
  for(int i=file_index;i<fs->FCB_next;i++){
    if(fs->FCB_table[i].file_type==1){
      update_parent(fs,i);
    }
  }
  // FCB move.
  for(int i=file_index;i<fs->FCB_next;i++){
    if(i+1!=fs->FCB_next){
      fs->FCB_table[i] = fs->FCB_table[i+1];
    }
  }
  fs->FCB_table[fs->FCB_next].start = -2;     
  fs->FCB_next--;
}

// Remove specific file
__device__ void remove_file(FileSystem *fs, int file_index){
  FCB_entry fe = fs->FCB_table[file_index];
  int old_block = fe.size/32 + (fe.size%32==0 ? 0:1);
  // data move.
  int start = fe.start + old_block*32;
  int end = fe.start;
  data_move(fs,start,end);
  // Update parent
  if(fe.parent!=-1){
    fs->FCB_table[fe.parent].size -= (my_strlen(fe.file_name)+1);  
  }
  for(int i=file_index;i<fs->FCB_next;i++){
    if(fs->FCB_table[i].file_type==1){
      update_parent(fs,i);
    }
  }
  // FCB move.
  for(int i=file_index;i<fs->FCB_next;i++){
    if(i+1!=fs->FCB_next){
      fs->FCB_table[i] = fs->FCB_table[i+1];
      if(fs->FCB_table[i].start!=-1) fs->FCB_table[i].start -= old_block*32;
    }
  }
  fs->FCB_table[fs->FCB_next].start = -2;     
  fs->FCB_next--; 
  // VCB modify.    
  fs->VCB_next-=old_block;   
}

__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
							int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
							int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE, 
							int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS)
{
  // init variables
  fs->volume = volume;

  // init constants
  fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
  fs->FCB_SIZE = FCB_SIZE;
  fs->FCB_ENTRIES = FCB_ENTRIES;
  fs->STORAGE_SIZE = VOLUME_SIZE;
  fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
  fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
  fs->MAX_FILE_NUM = MAX_FILE_NUM;
  fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
  fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;

  // init three blocks
  fs->VCB = volume;
  fs->FCB_table = (FCB_entry*)(volume + SUPERBLOCK_SIZE);
  fs->Content = volume + FILE_BASE_ADDRESS;
  fs->VCB_next = 0;
  fs->FCB_next = 0;
  fs->Content_next = 0;
  fs->current_time = 0;
  fs->target_FCB_entry = 0;
  fs->current_dir = -1;
  init_VCB(fs);

  // Remove
  fs->remove_file_count = 0;
  fs->remove_dir_count = 0;
}

__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
	/* Implement open operation here */
  fs->current_time++;
  int fp = name_search(fs,s,0,fs->current_dir);
  // Read part
  if (op==0){
    if(fp==-1){
      printf("ERROR: READ INEXISTENT FILE\n");
      return 0;
    }else{
      return fp;
    }
  }

  // Write part
  if (op==1){
    // Can not find, create a new FCB entry.
    if(fp==-1){
      FCB_entry fe;
      fe.size = 0;
      fe.start = fs->Content_next;
      fe.time = fs->current_time;
      fe.file_type = 0;
      fe.parent = fs->current_dir;
      deep_copy((char*)fe.file_name,s);
      fs->FCB_table[fs->FCB_next]=fe;
      fs->target_FCB_entry = fs->FCB_next;
      fs->FCB_next++;
      // Update the parent size
      fs->FCB_table[fs->current_dir].size += my_strlen(s)+1;
      return fe.start;
    }else{
      return fp;
    }
  }
  printf("ERROR: INEXISTENT OPERATION\n");
  return 0;

}

__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
	/* Implement read operation here */
  for(int i=0;i<size;i++){
    output[i] = fs->Content[fp+i];
  }
}

__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{
	/* Implement write operation here */

  fs->current_time++;
  int required_block = size/32+(size%32==0 ? 0:1);
  
  // If file is a new one.
  if(fp==fs->Content_next){
    // Write in Content.
    for(int i=0;i<size;i++){
      fs->Content[i+fp] = input[i];
    }
    fs->Content_next+=32*required_block;
    // Update VCB.
    for(int i=fp/32;i<required_block;i++){
      fs->VCB[i] = 1;
    }
    fs->VCB_next+=required_block;
    // Update FCB table.
    fs->FCB_table[fs->target_FCB_entry].start = fp;
    fs->FCB_table[fs->target_FCB_entry].size = size;
    fs->FCB_table[fs->target_FCB_entry].time = fs->current_time;
  }else{
  // if file is existent.
    FCB_entry fe = fs->FCB_table[fs->target_FCB_entry];
    int old_block = fe.size/32 + (fe.size%32==0 ? 0:1);
    // Same Block Number 
    if(old_block==required_block){
      // Write in new content and clean up.
      for(int i=0;i<size;i++){
        fs->Content[i+fp] = input[i];
      }
      if(size<fe.size){
        for(int i=fp+size;i<fp+fe.size;i++){
          fs->Content[i] = NULL;
        }
      }
      // Update FCB table
      fs->FCB_table[fs->target_FCB_entry].size = size;
      fs->FCB_table[fs->target_FCB_entry].time = fs->current_time;
    }
    // Old block number is different from new block number.
    else{
      // data move.
      int start = fe.start + old_block*32;
      int end = fe.start + required_block*32;
      data_move(fs,start,end);
      // Write in new content and clean up.
      for(int i=0;i<size;i++){
        fs->Content[i+fp] = input[i];
      }
      for(int i=fp+size;i<fp+required_block*32;i++){
        fs->Content[i] = NULL;
      }
      // FCB table modify.
      fs->FCB_table[fs->target_FCB_entry].size = size;
      fs->FCB_table[fs->target_FCB_entry].time = fs->current_time;
      // FCB table move.
      FCB_update(fs,start,end);
      // VCB modify.
      fs->VCB_next-=old_block-required_block;
    }
  }
  return 0;
}

__device__ void fs_gsys(FileSystem *fs, int op)
{
	/* Implement LS_D and LS_S operation here */
  int largest = ~(1<<31);
  int target = -1;
  int create_order = -1;
  int count = 0;
  int number = count_file_under_dir(fs);
  // LS_D
  if(op==0){
    printf("===sort by modifed time===\n");
    while(count<number){
      largest = find_largest_time(fs,largest);
      count++;
    }
  }
  // LS_S
  else if(op==1){
    printf("===sort by file size===\n");
    while(count<number){
      target = find_largest_size(fs,largest,create_order);
      largest = fs->FCB_table[target].size;
      create_order = target;
      count++;
    }  
  }
  // PWD
  else if(op==7){
    int dir = fs->current_dir;
    char file_names[3][20];
    char pwd[100];
    int count = 0;
    while(dir!=-1){
      deep_copy(file_names[count++],fs->FCB_table[dir].file_name);
      dir = fs->FCB_table[dir].parent;
    }
    deep_copy(pwd,"/");
    my_strcat(pwd,file_names[count-1]);
    if(count>=2){
      for (int i=count-2;i>=0;i--){
        my_strcat(pwd,"/");
        my_strcat(pwd,file_names[i]);
      }
    }
    printf("%s\n",pwd);
  }
  // CD_P
  else if(op==5){
    if (fs->current_dir==-1){
      printf("ERROR\n: THE DIR ARE IN THE TOP");
      return;
    }else{
      fs->current_dir = fs->FCB_table[fs->current_dir].parent;
    }
  } 

}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
	/* Implement rm operation here */
  // Remove
  if(op==2){
    int fp = name_search(fs,s,0,fs->current_dir);
    if (fp==-1){
      printf("ERROR:DELET INEXISTENT FILE\n");
      return;
    }
    else if(fs->FCB_table[fs->target_FCB_entry].file_type==1){
      printf("ERROR:DELET DIRECTORY\n");
      return;
    }
    else{
      FCB_entry fe = fs->FCB_table[fs->target_FCB_entry];
      int old_block = fe.size/32 + (fe.size%32==0 ? 0:1);
      // data move.
      int start = fe.start + old_block*32;
      int end = fe.start;
      data_move(fs,start,end);
      // Update parent
      if(fs->current_dir!=-1){
        fs->FCB_table[fs->current_dir].size -= (my_strlen(s)+1);  
      }
      for(int i=fs->target_FCB_entry;i<fs->FCB_next;i++){
        if(fs->FCB_table[i].file_type==1){
          update_parent(fs,i);
        }
      }
      // FCB move.
      for(int i=fs->target_FCB_entry;i<fs->FCB_next;i++){
        if(i+1!=fs->FCB_next){
          fs->FCB_table[i] = fs->FCB_table[i+1];
          if(fs->FCB_table[i].start!=-1) fs->FCB_table[i].start -= old_block*32;
        }
      }
      fs->FCB_table[fs->FCB_next].start = -2;     
      fs->FCB_next--; 
      // VCB modify.    
      fs->VCB_next-=old_block;
    }     
  }
  // MKDIR
  else if(op==3){
    // Create a new FCB entry
    fs->current_time++;
    FCB_entry fe;
    fe.start = -1;
    fe.time = fs->current_time;
    fe.size = 0;
    fe.file_type = 1;
    fe.parent = fs->current_dir;
    deep_copy(fe.file_name,s);
    fs->FCB_table[fs->FCB_next++] = fe;
    // Update the parent dir.
    if(fs->current_dir != -1){
      fs->FCB_table[fs->current_dir].size += my_strlen(s) + 1;
    } 
  }
  // CD
  else if(op==4){
    int fp = name_search(fs,s,1,fs->current_dir);
    if (fp==-1){
      printf("ERROR: CD TO INEXISTENT FILE\n");
      return;
    }else if(fs->FCB_table[fp].parent!=fs->current_dir){
      printf("ERROR: THE FILE IS NOT EXIST UNDER CURRENT DIR\n");
      return;
    }else{
      fs->current_dir = fp;
    }
  }
  // RM_RF
  else if(op==6){
    int fp = name_search(fs,s,1,fs->current_dir);
    fs->remove_dir[fs->remove_dir_count++] = fp;
    get_all_dir_and_file(fs,fp);
    for (int i=fs->remove_file_count-1;i>=0;i--){
      remove_file(fs,fs->remove_file[i]);
    }
    for (int i=fs->remove_dir_count-1;i>=0;i--){
      remove_dir(fs,fs->remove_dir[i]);
    }
    fs->remove_file_count = 0;
    fs->remove_dir_count = 0;
  }
  else{
    printf("ERROR:INVALID OP\n");
  }
}
