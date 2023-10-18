// nvcc -arch=sm_60 -run warp_cosinesim.cu cuda_helper.cu
#include "cuda_runtime.h"
#include "chTimer.h"
#include "cuda_helper.h"
#include <stdio.h>
#include "device_launch_parameters.h"
#include <cuda_profiler_api.h>
#define WARP_SIZE 32
#define NDIM 3
#define NPAR 2048*2048
#define NX 256
#define NY 128
#define NZ 128
#define NXY NX*NY
#define DX 0.01f
#define CELLSIZE DX*3.0f
#define PATM 0.0981f
#define NH DX*3.0f
#define NH2 NH*NH
#define KCOEF  21.f / (256.f * 3.14159f)

/*The objective of this exercise is to create a kernel, using shuffle instructions
to compute the dot product of many vectors. we will compare to a shared memory approach*/

/* One warp per dot product*/
__global__ void k_warp_densacc_fluid(float* all_positions,  //3dim
									 float* all_velocities,  //3dim
									 float* all_accelerations, //3dim
									 float* all_pressures,
									 float* all_masses,
									 float* all_densities,
									 float* all_dev_densities, 
									 float* all_du,
									 int* cell_starts,
									 int* cell_counts)
{
	int lid = threadIdx.x % WARP_SIZE;
	int wid = (blockIdx.x * blockDim.x + threadIdx.x);
	if(wid >= NPAR) return;
	int hash = wid;
	//find hash 
	//iterate all over all the ib in the cell
	float xa, ya, za, vxa, vya, vza, pa, ma, rhoa, dxa;
	// compute ia-ib distance if distance > 0 
	
		xa   = all_positions[wid * NDIM];
		ya   = all_positions[wid * NDIM + 1];
		za   = all_positions[wid * NDIM + 2]; 
		vxa  = all_velocities[wid * NDIM];  
		vya  = all_velocities[wid * NDIM + 1];  
		vza  = all_velocities[wid * NDIM + 2]; 
		pa   = all_pressures[wid] - PATM;
		ma   = all_masses[wid];
		rhoa = all_densities[wid];
	
	xa = __shfl_sync(0xffffffff, xa, 0); ya = __shfl_sync(0xffffffff, ya, 0); za = __shfl_sync(0xffffffff, za, 0);
	vxa = __shfl_sync(0xffffffff, vxa, 0);  vya = __shfl_sync(0xffffffff, vya, 0);   vza = __shfl_sync(0xffffffff, vza, 0);  
	pa = __shfl_sync(0xffffffff, pa, 0); ma = __shfl_sync(0xffffffff, ma, 0); rhoa = __shfl_sync(0xffffffff, rhoa, 0);
	dxa  = __powf(ma/rhoa, 1.0/3.0f);
	//cycle over all other particles
	int cellcounter = cell_counts[hash];
	int cellstart   = cell_starts[hash];
	float drho = 0.0f;
	
	float accx = 0.0f;
	float accy = 0.0f;
	float accz = 0.0f;
	//load my particle
	int ib = cellstart + lid; 
	float xb = all_positions[ib * NDIM];
	float yb = all_positions[ib * NDIM + 1];
	float zb = all_positions[ib * NDIM + 2]; 
	float xba = xa - xb;
	float yba = ya - yb;
	float zba = za - zb;
	float r2 = xba*xba + yba*yba + zba*zba;
	//float r2 = distance
	float pb = all_pressures[ib] - PATM;
	float mb = all_masses[ib];
	float rhob = all_densities[ib];
	float dxb = powf(mb/rhob, 1.0f/3.0f);
	float he = 0.5f * ( dxa + dxb);
	float pfactor = mb * ((pa+pb)/(rhoa*rhob));
		
	if (r2 < (NH2 * he * he)){
		float rba = sqrtf(r2);
		//if (rab < aux_rabmin) aux_rabmin=rab
		float vbax = vxa - all_velocities[ib * NDIM]; 
		float vbay = vya - all_velocities[ib * NDIM + 1];
		float vbaz = vza - all_velocities[ib * NDIM + 2];
		float dwe = 10.f * (rba/he) * powf(((rba-he) - 2.f), 3.f);// kernelderive(rba,he); fix
		float dwx = xba * dwe; 
		float dwy = yba * dwe; 
		float dwz = zba * dwe; 			
		float dume = vbax * dwx + vbay * dwy + vbaz * dwz;										
		drho += mb * dume;										
		accx += pfactor * dwx;
		accy += pfactor * dwy;
		accz += pfactor * dwz;
	}
	// Reductions & write to global memory
	for (int i=16; i>=1; i/=2){
        accx += __shfl_xor_sync(0xffffffff, accx, i, 32);
		accy += __shfl_xor_sync(0xffffffff, accy, i, 32);
		accz += __shfl_xor_sync(0xffffffff, accz, i, 32);
		drho += __shfl_xor_sync(0xffffffff, drho, i, 32);
	}
	
	// write to global memory
	
	all_accelerations[wid * NDIM] 	 = accx;
	all_accelerations[wid * NDIM + 1] = accy;
	all_accelerations[wid * NDIM + 2] = accz;
	all_dev_densities[wid] = drho;
	all_du[wid] = 0.5f * pa / (rhoa * rhoa) * drho;

}




/* One thread per dot product*/
__global__ void k_naive_densacc_fluid(float* all_positions,  //3dim
									  float* all_velocities,  //3dim
									  float* all_accelerations, //3dim
									  float* all_pressures,
									  float* all_masses,
									  float* all_densities,
									  float* all_dev_densities, 
									  float* all_du,
									  int* cell_starts,
									  int* cell_counts)
{
	int ia = blockIdx.x * blockDim.x + threadIdx.x;
	if(ia >= NPAR) return;
	int hash = ia/WARP_SIZE; // eqv ia / WARP_SIZE;// for testing purpouses eqv to wid
	//find hash 
	//iterate all over all the ib in the cell
	// compute ia-ib distance if distance > 0 
	float xa   = all_positions[ia * NDIM];
	float ya   = all_positions[ia * NDIM + 1];
	float za   = all_positions[ia * NDIM + 2]; 
	float vxa  = all_velocities[ia * NDIM];  
	float vya  = all_velocities[ia * NDIM + 1];  
	float vza  = all_velocities[ia * NDIM + 2]; 
	float pa   = all_pressures[ia] - PATM;
	float ma   = all_masses[ia];
	float rhoa = all_densities[ia];
	float dxa  = powf(ma/rhoa, 1.0f/3.0f);
	//cycle over all other particles
	int cellcounter = cell_counts[hash];
	int cellstart   = cell_starts[hash];
	float drho = 0.0f;
	
	float accx = 0.0f;
	float accy = 0.0f;
	float accz = 0.0f;
	
	for(int lid = 0; lid < cellcounter; lid++){
		//load
		int ib = cellstart + lid; 
		float xb = all_positions[ib * NDIM];
		float yb = all_positions[ib * NDIM + 1];
		float zb = all_positions[ib * NDIM + 2]; 
		float xba = xa - xb;
		float yba = ya - yb;
		float zba = za - zb;
		float r2 = xba*xba + yba*yba + zba*zba;
		//float r2 = distance
		float pb = all_pressures[ib] - PATM;
		float mb = all_masses[ib];
		float rhob = all_densities[ib];
		float dxb = powf(mb/rhob, 1.0/3.0f);
		float he = 0.5f * ( dxa + dxb);
		float pfactor = mb * ((pa+pb)/(rhoa*rhob));
		
		if (r2 < (NH2 * he * he)){
			float rba = sqrtf(r2);
			//if (rab < aux_rabmin) aux_rabmin=rab
			float vbax = vxa - all_velocities[ib * NDIM]; 
			float vbay = vya - all_velocities[ib * NDIM + 1];
			float vbaz = vza - all_velocities[ib * NDIM + 2];
			float dwe = 10.f * (rba/he) * powf(((rba-he) - 2.f), 3.f);// kernelderive(rba,he); fix
			float dwx = xba * dwe; 
			float dwy = yba * dwe; 
			float dwz = zba * dwe; 			
			float dume = vbax * dwx + vbay * dwy + vbaz * dwz;										
			drho += mb * dume;										
			accx += pfactor * dwx;
			accy += pfactor * dwy;
			accz += pfactor * dwz;
		}
	} 
			
	// write to global memory
   	all_accelerations[ia * NDIM] 	 = accx;
	all_accelerations[ia * NDIM + 1] = accy;
	all_accelerations[ia * NDIM + 2] = accz;
	all_dev_densities[ia] = drho;
	all_du[ia] = 0.5f * pa / (rhoa * rhoa) * drho;
}


/////////////////////////////////Diagnostic routines/////////////////////////////////////////////
int check_equal_float_vec(float *vec1,float *vec2,int size){
	int numerrors = 0;
	float dist;
	float tolerance = 0.0001f;
	for(int i =0; i< size; i++){
	    dist = (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
		if(dist > tolerance) numerrors++;
	}
	if(numerrors ==0)printf("Congratulations you have 0 errors!\n");
	if(numerrors >0)printf("Wrong results, you have %d errors!\n", numerrors);

	return numerrors;
}
struct cxyz_t{
	public:
		int cx; 
		int cy; 
		int cz;
};
\
__device__ float kernelderive (float rba, float he) {
	float q = rba / he;
	if (q < 2.f) {
		return 10.f * q * powf((q - 2.f), 3.f);
	} else { 
		return 0.f;
	}
}
__device__ __host__ cxyz_t hashToCells (int hasvalue) {
	cxyz_t mycell;
	mycell.cz = hasvalue / (NXY);
	int cxy = hasvalue % NXY;
	mycell.cy = cxy / NX;
	mycell.cx = cxy % NX;
	
	return mycell;
}

__device__ __host__ int cellidsToHash (int cx, int cy, int cz) {
	int myhash = cx + cy * NX + cz * NXY;
	return myhash;
}

__device__ __host__ int cellToHash (cxyz_t cell) {
	int myhash = cell.cx + cell.cy * NX + cell.cz * NXY;
	return myhash;
}
//////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	cudaClock cknaive, ckshmem, ckwarp;
	// host 
	float *h_positions, *h_velocities, *h_accelerations; 
	float *h_density, *h_dev_density, *h_pressure, *h_du, *h_mass;
	int *h_cell_count, *h_cell_starts;//, *h_cell_particles;
	int *h_particle_hash, *h_particle_pos;
	// device
	float *d_positions, *d_velocities, *d_accelerations;
	float *d_density, *d_dev_density, *d_pressure, *d_du, *d_mass;
	int *d_cell_count, *d_cell_starts;//, *h_cell_particles;
	int *d_particle_hash, *d_particle_pos;
	printf("\nStarting program execution..\n\n");
	
	printf("\nAllocating and creating problem data..\n");
	// float ones
	h_positions     = (float*) malloc(NPAR * NDIM * sizeof(float));
	h_velocities    = (float*) malloc(NPAR * NDIM * sizeof(float));
	h_accelerations = (float*) malloc(NPAR * NDIM * sizeof(float));
	h_density		= (float*) malloc(NPAR * sizeof(float));
	h_dev_density   = (float*) malloc(NPAR * sizeof(float));
	h_pressure		= (float*) malloc(NPAR * sizeof(float));
	h_du			= (float*) malloc(NPAR * sizeof(float));
	h_mass			= (float*) malloc(NPAR * sizeof(float));
	// int ones 
	h_cell_count    = (int*)malloc(NXY * NZ * sizeof(int));
	h_cell_starts   = (int*)malloc(NXY * NZ * sizeof(int));
	h_particle_hash = (int*)malloc(NPAR * sizeof(int));
	h_particle_pos  = (int*)malloc(NPAR * sizeof(int));


	for(int ih = 0; ih < NXY * NZ; ih++){
		int wid = ih / WARP_SIZE;
		if(ih <= NPAR/WARP_SIZE) {
			h_cell_count[ih]  = WARP_SIZE;
			h_cell_starts[ih] = WARP_SIZE * ih;
		} else {
			h_cell_count[ih]  = 0;
			h_cell_starts[ih] = NPAR;
		}
	}
	
	for(int ip = 0; ip < NPAR; ip++){
		int wid = ip / WARP_SIZE;
		cxyz_t my_cell = hashToCells(wid);
		float cell_origen_x = my_cell.cx * CELLSIZE;
		float cell_origen_y = my_cell.cy * CELLSIZE;
		float cell_origen_z = my_cell.cz * CELLSIZE;
		int lid = ip % WARP_SIZE;
		int zstep = lid / 8;
		int xystep = lid % 9;
		int xstep = xystep / 3;
		int ystep = xystep % 3;
		float xlocal = xstep * 0.01f + 0.005f;
		float ylocal = ystep * 0.01f + 0.005f;
		float zlocal = zstep * 0.0075f + 0.0035f;
		h_positions[ip*3] 	      = xlocal + cell_origen_x; //x
		h_positions[ip*3 + 1]     = ylocal + cell_origen_y; //y
		h_positions[ip*3 + 2]     = zlocal + cell_origen_z; //z
		h_velocities[ip*3] 	      = 0.0f; //vx
		h_velocities[ip*3 + 1]    = 0.0f; //vy
		h_velocities[ip*3 + 2]    = 0.0f; //vz
		h_accelerations[ip*3] 	  = 0.0f; //ax
		h_accelerations[ip*3 + 1] = 0.0f; //ay
		h_accelerations[ip*3 + 2] = 0.0f; //az
		h_density[ip] 		  	  = 0.027f; //d
		h_dev_density[ip] 	 	  = 0.0f; //d
		h_pressure[ip] 		  	  = (NZ - my_cell.cz) * CELLSIZE * 0.098f; //pz
		h_du[ip]				  = 0.0f;
		h_mass[ip]				  = 0.001f;
		//
		h_particle_hash[ip] = wid;
		h_particle_pos[ip] = lid;
	}
	 ////
	 //------ Step 1: Allocate the memory-------
	 printf("\nAllocating Device Memory\n");
	CudaSafeCall(cudaMalloc((void**)&d_positions,     NPAR * NDIM * sizeof(float)));
	CudaSafeCall(cudaMalloc((void**)&d_velocities,    NPAR * NDIM * sizeof(float)));
	CudaSafeCall(cudaMalloc((void**)&d_accelerations, NPAR * NDIM * sizeof(float)));
	CudaSafeCall(cudaMalloc((void**)&d_density,       NPAR * sizeof(float)));
	CudaSafeCall(cudaMalloc((void**)&d_dev_density,   NPAR * sizeof(float)));
	CudaSafeCall(cudaMalloc((void**)&d_pressure,      NPAR * sizeof(float)));
	CudaSafeCall(cudaMalloc((void**)&d_du,      	  NPAR * sizeof(float)));
	CudaSafeCall(cudaMalloc((void**)&d_mass,      	  NPAR * sizeof(float)));
	CudaSafeCall(cudaMalloc((void**)&d_cell_count,    NXY * NZ * sizeof(int)));
	CudaSafeCall(cudaMalloc((void**)&d_cell_starts,   NXY * NZ * sizeof(int)));
	CudaSafeCall(cudaMalloc((void**)&d_particle_hash, NPAR * sizeof(int)));
	CudaSafeCall(cudaMalloc((void**)&d_particle_pos,  NPAR * sizeof(int)));

	checkGPUMemory();

	//------ Step 2: Copy Memory to the device-------
	printf("\nTransfering data to the Device..\n");
	CudaSafeCall(cudaMemcpy(d_positions, h_positions, 	  NPAR * NDIM * sizeof(float), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_velocities, h_velocities,   NPAR * NDIM * sizeof(float), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_density, h_density, 		  NPAR * sizeof(float), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_pressure, h_pressure, 	  NPAR * sizeof(float), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_mass, h_mass, 			  NPAR * sizeof(float), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_cell_count, h_cell_count,   NXY * NZ * sizeof(int), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_cell_starts, h_cell_starts, NXY * NZ * sizeof(int), cudaMemcpyHostToDevice));

	CudaCheckError();
	//------ Step 3: Prepare launch parameters-------
	printf("\npreparing launch parameters..\n");
	//warp launch params .. check this
	dim3 warpGrid = dim3((NPAR + 127)/128, 1, 1);
	dim3 warpBlock = dim3(128,1,1);

	//naive GPU launch params
	dim3 naiveGrid = dim3((NPAR + 127)/128, 1, 1);//
	dim3 naiveBlock = dim3(128,1,1);
	//------ Step 4: Launch device kernel-------
	printf("\nLaunch Device Kernels.\n");

	CudaCheckError();
	// KERNEL LAUNCHS GOES HERE------------------------>>>>>>>>>
	printf("\nLaunch Naive Kernel.\n");
	cudaProfilerStart();
	cudaTick(&cknaive);
	
	k_naive_densacc_fluid<<<naiveGrid, naiveBlock>>>(d_positions,
													 d_velocities,
													 d_accelerations,
													 d_pressure,
													 d_mass,
													 d_density,
													 d_dev_density, 
													 d_du,
													 d_cell_starts,
													 d_cell_count);
	cudaTock(&cknaive, "naive kernel");
	//CudaCheckError();
	//cudaDeviceSynchronize();
	printf("\nLaunch Warp Kernel.\n");
	cudaTick(&ckwarp);
	k_warp_densacc_fluid<<<warpGrid, warpBlock>>>(d_positions,
												  d_velocities,
												  d_accelerations,
												  d_pressure,
												  d_mass,
												  d_density,
												  d_dev_density, 
												  d_du,
											      d_cell_starts,
												  d_cell_count);
	cudaTock(&ckwarp, "warp shuffle kernel");
	cudaProfilerStop();
	std::cout << std::endl;
	//CudaCheckError();
	//------ Step 5: Copy Memory back to the host-------
	printf("\nTransfering result data to the Host..\n");
	// CudaSafeCall(cudaMemcpy(h_scores, d_scores, DICTIONARY_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
	//
	printf("\nComparing  times..\n");

	std::cout << "the warp shuffle kernel is " << cknaive.elapsedMicroseconds/ckwarp.elapsedMicroseconds << " times faster than naive kernel" << std::endl;


	// -----------Step 6: Free the memory --------------
	printf("Deallocating device memory..\n");

	cudaFree(d_positions);
	cudaFree(d_velocities);
	cudaFree(d_accelerations);
	cudaFree(d_density);
	cudaFree(d_dev_density);
	cudaFree(d_pressure);
	cudaFree(d_du);
	cudaFree(d_mass);
	cudaFree(d_cell_count);
	cudaFree(d_cell_starts);
	cudaFree(d_particle_hash);
	cudaFree(d_particle_pos);

	printf("Deallocating host memory..\n");
	free(h_positions);
	free(h_velocities);
	free(h_accelerations);
	free(h_density);
	free(h_dev_density);
	free(h_pressure);
	free(h_du);
	free(h_mass);
	free(h_cell_count);
	free(h_cell_starts);
	free(h_particle_hash);
	free(h_particle_pos);

	return 0;
}

