// Primer Ejemplo de una operación MAP
#include <stdio.h>
#include <windows.h>

#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

//Maximo prefix sum scan, reduccion


//Definición de variables globales (SIZE, etc)

#define VECTOR_SIZE 1024 * 1024 //* 128
#define BLOCK_SIZE 256


int reduccionCPU(float* vector,
                 int vector_size)
{

    float sum = 0.0f;

    for (int i=0; i< vector_size; i++)
        sum += vector[i];

    return sum;
}

void prefixMaxScanCPU(float* input,
                     float* output,
                     int vector_size){

    output[0] = 0.0f;
    for(int i=1; i<=vector_size; i++){
        output[i] = max(input[i-1], output[i-1]);
    }
}

//diagnóstico
int compare_vectors(float *A, float *B, int size){

	int discrepancies = 0;
    float tolerance = 20.0f;
	for(int i = 0; i < size; i++){
	   float dist = (A[i] - B[i])*(A[i] - B[i]);
       if(dist > tolerance){
           discrepancies++;
           printf("cpu %.03f gpu %.03f\n", A[i], B[i]);
       }
	}
	
	if(discrepancies > 0){ 
		printf("\nHay %d diferencias entre ambos vectores!\n", discrepancies);
	} else {
		printf("\nLos dos vectores son clavaos!\n");
	}
	return discrepancies;
}

//ARRANQUE DE LA EJECUCION DEL PROGRAMA

int main(int argc,char **argv)
{

 cudaDeviceReset();
//configuramos la medida de tiempo de la CPU.
  LARGE_INTEGER ticksPerSecond;
  LARGE_INTEGER tick;  
  LARGE_INTEGER tock;

 // get the high resolution counter's accuracy
 QueryPerformanceFrequency(&ticksPerSecond);
 ////////////////////////////////////////////////////////////////////////////////////////
 ////////////////////////////////////////////////////////////////////////////////////////
 printf("Empezamos el programa\n");

 printf("Problem Size:vector de %d elementos\n", VECTOR_SIZE);
 
 
 //Declaración de variables
 
 float *h_input_reduce, h_output_gpu_reduce, h_output_cpu_reduce;
 float *h_input_prefix, *h_output_gpu_prefix, *h_output_cpu_prefix;

 float *d_input_reduce;

 float *d_input_prefix, *d_output_prefix;

 int size_bytes = VECTOR_SIZE * sizeof(float);

 printf("Reservamos memoria en el host...");

 h_input_reduce = (float*) malloc(size_bytes);
 h_input_prefix = (float*) malloc(size_bytes);
 h_output_cpu_prefix = (float*) malloc(size_bytes);
 h_output_gpu_prefix = (float*) malloc(size_bytes);
 printf("ok!\n");
 
 //inicialización del ejemplo
 printf("Inicializamos los vectores de prueba...");
 
 for (int i = 0; i < VECTOR_SIZE; i++){
     h_input_reduce[i] = 1;//rand() * 10.0f;
     h_input_prefix[i] = rand();
 }
 
 printf("ok!\n");
 
 //PASO 1: Reservamos la memoria en la GPU
 printf("PASO 1: Reservando memoria en la GPU...");

 cudaMalloc((void**) &d_input_reduce, size_bytes);
 cudaMalloc((void**) &d_input_prefix, size_bytes);
 cudaMalloc((void**) &d_output_prefix, size_bytes);
 
 printf("ok!\n");
 
 printf("PASO 2: Copiando memoria desde el host al device...");

 cudaMemcpy(d_input_reduce, h_input_reduce, size_bytes, cudaMemcpyHostToDevice);
 cudaMemcpy(d_input_prefix, h_input_prefix, size_bytes, cudaMemcpyHostToDevice);

 if(cudaSuccess != cudaGetLastError()) printf("Error in cudaMemCpy..not ");
 printf("ok!\n");



 
 cudaEvent_t gstart,gstop;//no os fijéis en estas líneas
 cudaEventCreate(&gstart);//nada importante
 cudaEventCreate(&gstop);//nada que ver...continuen..
 
 //PASO 4: Lanzamos el kernel
 printf("PASO 4: Lanzamos el kernel...");
 cudaEventRecord(gstart, 0);//nada importante...

 thrust::maximum<int> binary_op_max;

 thrust::exclusive_scan(thrust::device,
                        d_input_prefix,
                        d_input_prefix + VECTOR_SIZE, //N ELEMENTS
                        d_output_prefix,
                        0,
                        binary_op_max);

 cudaEventRecord(gstop, 0);//lalalalalala
 cudaEventSynchronize(gstop);//sigue sin importar
 if(cudaSuccess != cudaGetLastError()) printf("Error in GPU calculation...not ");
 printf("ok!\n\n");

 
 float gpu_time; 
 cudaEventElapsedTime(&gpu_time, gstart, gstop);
 printf("La version GPU ha necesitado %.3f milisegundos\n\n",gpu_time );
 
 cudaEventDestroy(gstart); //limpiando un pelín
 cudaEventDestroy(gstop); //insisto..nada que ver..circulen
 
 //PASO 5: Copiamos la memoria del host al device
 printf("PASO 5: Copiamos el resultado desde el device al host...");

 cudaMemcpy(h_output_gpu_prefix,d_output_prefix,size_bytes,cudaMemcpyDeviceToHost);

 printf("ok!\n");
 
 //PASO 6: Liberamos la memoria en la GPU
 printf("PASO 6: Liberando memoria de la GPU...");

 cudaFree(d_input_prefix);
 cudaFree(d_output_prefix);

 printf("ok!\n\n");
 
 printf("Ahora usamos la version secuencial mas que nada por comparar\n");
 QueryPerformanceCounter(&tick);

 prefixSumScanCPU(h_input_prefix,h_output_cpu_prefix, VECTOR_SIZE);

 QueryPerformanceCounter(&tock);
 double mtime = (tock.QuadPart - tick.QuadPart)*1000/(ticksPerSecond.QuadPart);
 printf("La version CPU ha necesitado %.3f milisegundos\n", mtime);
 
 printf("Lo que nos da un %.3f x \n",mtime/gpu_time);
 
 printf("Comparamos resultados\n");

 int errorcillos = compare_vectors(h_output_gpu_prefix,h_output_cpu_prefix,VECTOR_SIZE);


 printf("Y ahora la reducción");

 cudaEventCreate(&gstart);//nada importante
 cudaEventCreate(&gstop);//nada que ver...continuen..

 cudaEventRecord(gstart, 0);//nada importante...

 thrust::plus<int> binary_op_plus;
 h_output_gpu_reduce = thrust::reduce(thrust::device,
                                      d_input_reduce,
                                      d_input_reduce + VECTOR_SIZE,
                                      0,
                                      binary_op_plus);

 cudaEventRecord(gstop, 0);//lalalalalala
 cudaEventSynchronize(gstop);//sigue sin importar
 if(cudaSuccess != cudaGetLastError()) printf("Error in GPU calculation...not ");
 printf("ok!\n\n");


 cudaEventElapsedTime(&gpu_time, gstart, gstop);
 printf("La version GPU ha necesitado %.3f milisegundos\n\n",gpu_time );

 cudaEventDestroy(gstart); //limpiando un pelín
 cudaEventDestroy(gstop); //insisto..nada que ver..circulen

 printf("Ahora usamos la version secuencial mas que nada por comparar\n");
 QueryPerformanceCounter(&tick);

 h_output_cpu_reduce = reduccionCPU(h_input_reduce,VECTOR_SIZE);

 QueryPerformanceCounter(&tock);
 mtime = (tock.QuadPart - tick.QuadPart)*1000/(ticksPerSecond.QuadPart);
 printf("La version CPU ha necesitado %.3f milisegundos\n", mtime);

 printf("Lo que nos da un %.3f x \n",mtime/gpu_time);

 printf("Comparamos resultados cpu = %.05f gpu = %.05f\n", h_output_cpu_reduce, h_output_gpu_reduce);

 cudaFree(d_input_reduce);


 printf("Liberando memoria de la CPU...");
 free(h_input_prefix);
 free(h_input_reduce);
 free(h_output_gpu_prefix);
 free(h_output_cpu_prefix);
 printf("ok!\n");
 
 return 1;
}

