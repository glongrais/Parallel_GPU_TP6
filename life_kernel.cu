// 1512/1474
// Reads a cell at (x+dx, y+dy)
__device__ int read_cell(int * source_domain, int x, int y, int dx, int dy,
    unsigned int domain_x, unsigned int domain_y)
{
    x = (unsigned int)(x + dx) % domain_x;	// Wrap around
    y = (unsigned int)(y + dy) % domain_y;
    return source_domain[y * domain_x + x];
}

// Compute kernel
__global__ void life_kernel(int * source_domain, int * dest_domain,
    int domain_x, int domain_y)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(tx >= domain_x || ty >= domain_y) return;

    extern __shared__ int data_block[];

    data_block[(threadIdx.y+1) * (blockDim.x +2) + (threadIdx.x+1)] = read_cell(source_domain, tx, ty, 0, 0,
        domain_x, domain_y);

    if(threadIdx.y == 0){
        data_block[threadIdx.x+1] = read_cell(source_domain, tx, ty, 0, -1,
            domain_x, domain_y);
    }
    if(threadIdx.x == 0){
        data_block[(threadIdx.y+1) * (blockDim.x +2)] = read_cell(source_domain, tx, ty, -1, 0,
            domain_x, domain_y);
    }
    if(threadIdx.y == blockDim.y-1){
        data_block[(threadIdx.y+2) * (blockDim.x +2) + (threadIdx.x+1)] = read_cell(source_domain, tx, ty, 0, +1,
            domain_x, domain_y);
    }
    if(threadIdx.x == blockDim.x-1){
        data_block[(threadIdx.y+1) * (blockDim.x +2) + (threadIdx.x+2)] = read_cell(source_domain, tx, ty, +1, 0,
            domain_x, domain_y);
    }
    if(threadIdx.y == 0 && threadIdx.x == 0){
        data_block[0] = read_cell(source_domain, tx, ty, -1, -1,
            domain_x, domain_y);
    }
    if(threadIdx.y == 0 && threadIdx.x == blockDim.x-1){
        data_block[threadIdx.x+2] = read_cell(source_domain, tx, ty, -1, +1,
            domain_x, domain_y);
    }
    if(threadIdx.y == blockDim.y-1 && threadIdx.x == 0){
        data_block[(threadIdx.y+2) * (blockDim.x +2)] = read_cell(source_domain, tx, ty, +1, -1,
            domain_x, domain_y);
    }
    if(threadIdx.y == blockDim.y-1 && threadIdx.x == blockDim.x-1){
        data_block[(threadIdx.y+2) * (blockDim.x +2) + (threadIdx.x+2)] = read_cell(source_domain, tx, ty, +1, +1,
            domain_x, domain_y);
    }

    __syncthreads();


    // Read cell
    int myself = read_cell(data_block, threadIdx.x+1, threadIdx.y+1, 0, 0, blockDim.x+2, blockDim.y+2);
    //int myself = read_cell(source_domain, tx, ty, 0, 0, domain_x, domain_y);
    
    // TODO: Read the 8 neighbors and count number of blue and red
    int blue = 0, red = 0;
    for(int i = -1; i < 2; i++){
        for(int j = -1; j < 2; j++){
            if(i != 0 && j !=0){
                int cell = read_cell(data_block, threadIdx.x+1, threadIdx.y+1, i, j, blockDim.x+2, blockDim.y+2);
                //int cell = read_cell(source_domain, tx, ty, i, j, domain_x, domain_y);
                if(cell == 1){
                    red++;
                }else if(cell == 2){
                    blue++;
                }
            }
        }
    }
    // TODO: Compute new value
    
    if(myself == 0 && (blue + red) == 3){
        if(red >= 2){
            myself = 1; 
        }else{
            myself = 2;
        }
    }else if((blue + red) < 2 || (blue + red) > 3){
        myself = 0;
    }
	
    // TODO: Write it in dest_domain
    
    int x = (unsigned int)tx % domain_x;
    int y = (unsigned int)ty % domain_y;

    dest_domain[y * domain_x + x] = myself;
}

