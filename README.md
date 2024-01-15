# GPU-Based File System README

## Overview

This project implements a simplified file system within a GPU environment using CUDA. It's designed to emulate basic file system functionalities in the absence of an operating system, utilizing global memory as a logical drive. You can find the full project report [here](./report.pdf).

![Whole Picture](./figures/Whole%20Picture.jpg)

## Key Features

- **Simplified File System Structure**: Emulates a basic file system with a single root directory, without subdirectories.
- **Basic File Operations**: Supports fundamental file operations within the GPU memory.
- **Single-threaded Management**: Operates under a single thread, focusing on direct data management in global memory.
- **Tree-structured directories**: Organizes files in a tree structure. (Bonus)
![Tree-structured directories](./figures/tree-%20structure.jpg)

## Key Components
Below is the basic structure for a single-root directory.
![basic structure](./figures/basic%20structure.jpg)

- **VCB (0 to 4KB)**: Acts as a bitmap to indicate the availability of storage blocks.
- **FCB Table (4KB to ~36KB)**: Stores file metadata in 1024 entries, each 32B in size, including details like file name, start position, size, and modification time.
- **Storage (~36KB onwards)**: Used for storing file content, divided into 32B blocks.

Below is the bonus structure for tree-structured directories.
![bonus structure](./figures/bonus%20structure.jpg)

## Implementation Details
- **Global Memory as Volume**: Utilizes GPU's global memory to simulate a volume (logical drive).
- **Metadata Management**: Handles metadata information and file-control blocks (FCBs) to maintain file attributes and locations.
- **Direct Information Access**: Bypasses complex memory structures, directly accessing data from the volume.

## Specifications
- The size of volume is 1085440 bytes (1060KB). The size of files in total is 1048576 bytes (1024KB). The maximum number of file is 1024.
- The maximum size of a file is 1024 bytes (1KB). The maximum size of a file name is 20 bytes.
- File name end with “\0”.
- FCB size is 32 bytes.
- FCB entries is 32KB/ 32 bytes = 1024.
- Storage block size is 32 bytes.

## Functions

### `fs_open`
- **Purpose**: Open a file.
- **Parameters**: 
  - `FileSystem *fs`: File system pointer.
  - `char *s`: File name.
  - `int op`: Operation mode (G_READ or G_WRITE).
- **Returns**: Write/read pointer.
- **Usage**:
  ```c
  __device__ u32 fs_open(FileSystem *fs, char *s, int op);
  ```

### `fs_write`
- **Purpose**: Write to a file.
- **Parameters**: 
  - `FileSystem *fs`: File system pointer.
  - `uchar* input`: Input buffer.
  - `u32 size`: Size of the data (in bytes).
  - `u32 fp`: Write pointer.
- **Usage**:
  ```c
  __device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp);
  ```

### `fs_read`
- **Purpose**: Read contents from a file.
- **Parameters**: 
  - `FileSystem *fs`: File system pointer.
  - `uchar *output`: Output buffer.
  - `u32 size`: Size of the data (in bytes).
  - `u32 fp`: Read pointer.
- **Usage**:
  ```c
  __device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp);
  ```

### `fs_gsys` (RM)
- **Purpose**: Delete a file and release file space.
- **Parameters**:
  - `FileSystem *fs`: File system pointer.
  - `int op`: Command (RM denotes DELETE command).
  - `char *s`: File name.
- **Usage**:
  ```c
  __device__ void fs_gsys(FileSystem *fs, int op, char *s);
  ```

### `fs_gsys` (LS_D / LS_S)
- **Purpose**: List information about files.
- **Parameters**:
  - `FileSystem *fs`: File system pointer.
  - `int op`: Command (LS_D/LS_S for listing files).
- **Usage**:
  ```c
  fs_gsys(fs, LS_S);
  fs_gsys(fs, LS_D);
  ```

## Setup and Usage

- **Environment Requirements**: You can find the required environment setup in the [project report](./report.pdf).
- **Compilation and Running Instructions**:
```
// Basic version
cd source
sbatch slurm.sh

// Tree-structured version
cd bonus
sbatch slurm.sh
```