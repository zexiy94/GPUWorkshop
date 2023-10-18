size_t free_byte0 ;
size_t total_byte0 ;

 if ( cudaSuccess != cudaMemGetInfo( &free_byte0, &total_byte0 ) ){
      printf("Error: cudaMemGetInfo fails \n" );
      exit(1);
    }

double free_db0 = ((double)free_byte0)/(1024.0*1024.0) ;
double total_db0 = ((double)total_byte0)/(1024.0*1024.0) ;
 double used_db0 = total_db0 - free_db0 ;

printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",used_db0, free_db0, total_db0);