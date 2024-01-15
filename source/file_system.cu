#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

// Search the file in the FCB table
__device__ int name_seach(FileSystem *fs, char *s){
  for(int i=0;i<fs->FCB_next;i++){
    FCB_entry fe = fs->FCB_table[i];
    if(name_compare(fe.file_name,s)){
      fs->target_FCB_entry = i;
      return fe.start;
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
    fs->FCB_table[i].start+=end-start;
  }
}

// Find the largest modified time
__device__ int find_largest_time(FileSystem *fs, int prev_largest){
  int largest = -1;
  int target = -1;
  for(int i=0;i<fs->FCB_next;i++){
    int time = fs->FCB_table[i].time;
    if (time < prev_largest && time > largest){
        largest = time;
        target = i;
    }
  }
  printf("%s\n",fs->FCB_table[target].file_name);
  return largest;
}

// Find the largest file size
__device__ int find_largest_size(FileSystem *fs, int prev_largest, int create_order){
  int largest = -1;
  int target = -1;
  for(int i=0;i<fs->FCB_next;i++){
    int size = fs->FCB_table[i].size;
    if (size == prev_largest && i>create_order){
      target = i;
      break;
    }else if(size<prev_largest && size>largest){
      largest=size;
      target = i;
    }
  }
  printf("%s %d\n",fs->FCB_table[target].file_name,fs->FCB_table[target].size); 
  return target;
}

__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
							int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
							int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE, 
							int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS){
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
  init_VCB(fs);

}

__device__ u32 fs_open(FileSystem *fs, char *s, int op){
	/* Implement open operation here */
  fs->current_time++;
  int fp = name_seach(fs,s);
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
      deep_copy((char*)fe.file_name,s);
      fs->FCB_table[fs->FCB_next]=fe;
      fs->target_FCB_entry = fs->FCB_next;
      fs->FCB_next++;
      return fe.start;
    }else{
      return fp;
    }
  }
  printf("ERROR: INEXISTENT OPERATION\n");
  return 0;

}

__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp){
	/* Implement read operation here */
  for(int i=0;i<size;i++){
    output[i] = fs->Content[fp+i];
  }
}

__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp){
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

__device__ void fs_gsys(FileSystem *fs, int op){
	/* Implement LS_D and LS_S operation here */
  int largest = ~(1<<31);
  int target = -1;
  int create_order = -1;
  int count = 0;
  // LS_D
  if(op==0){
    printf("===sort by modifed time===\n");
    while(count<fs->FCB_next){
      largest = find_largest_time(fs,largest);
      count++;
    }
  }else if(op==1){
    printf("===sort by file size===\n");
    while(count<fs->FCB_next){
      target = find_largest_size(fs,largest,create_order);
      largest = fs->FCB_table[target].size;
      create_order = target;
      count++;
    }  
  }
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s){
	/* Implement rm operation here */
  if(op==2){
    int fp = name_seach(fs,s);
    if (fp==-1){
      printf("ERROR:DELET INEXISTENT FILE\n");
      return;
    }else{
      FCB_entry fe = fs->FCB_table[fs->target_FCB_entry];
      int old_block = fe.size/32 + (fe.size%32==0 ? 0:1);
      // data move.
      int start = fe.start + old_block*32;
      int end = fe.start;
      data_move(fs,start,end);
      // FCB move.
      for(int i=fs->target_FCB_entry;i<fs->FCB_next;i++){
        if(i+1!=fs->FCB_next){
          fs->FCB_table[i] = fs->FCB_table[i+1];
          fs->FCB_table[i].start -= old_block*32;
        }
      }      
      fs->FCB_next--; 
      // VCB modify.    
      fs->VCB_next-=old_block;
    }     
  }else{
    printf("ERROR:INVALID OP\n");
  }
}
