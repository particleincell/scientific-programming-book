/*Distributed Computing Lesson 8 

Example of using OpenGL to perform visualization 
of GPU computed results

step 3: uses CUDA to generate data, which is then copied to the CPUs
*/

// OpenGL Graphics includes
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <iostream>

// CUDA includes
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

using namespace std;

#define CUDA_ERROR(f) {if (f!=cudaSuccess) {cerr<<cudaGetErrorString(f)<<" on line "<<__LINE__<<endl;exit(-1);}}

#define DIM 500
#define PI 3.14159265359 

static void key_func(unsigned char key, int x, int y)
{
	switch(key)
	{
	case 27: /*esc*/
		exit(0);
	}
}

GLuint buffer_obj;
GLubyte *rgba_data;
GLubyte *dev_data;

static void draw_func(void)
{
	glDrawPixels(DIM, DIM, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glutSwapBuffers();
}

/*some placeholder function to generate data*/
__global__ void gen_data_gpu(GLubyte *data, float t)
{
	int i = blockIdx.x;
	int j = blockIdx.y;
	
	int o = (j*DIM+i)*4;
	float fi = i/(float)(DIM-1.0);	/*convert to [0,1]*/
	float fj = j/(float)(DIM-1.0);
	float val = cos(fi*2*PI+t)*cos(t*fj*PI);
	float z = (val+1);	/*convert values to [0,2]*/
	int r,g;
	if (z<1.0) {r=z*255;g=0;}
	else {r=255;g=(z-1)*255;}
	data[o+0] = r;
	data[o+1] = g;
	data[o+2] = 0;
	data[o+3] = 255;
}

//uses CUDA to update the function values
void generateData()
{
	static float t = 0;
	static float dt = 0.1;

	dim3 dims = {DIM,DIM,1};
	gen_data_gpu<<<dims,1>>>(dev_data,t);
	CUDA_ERROR(cudaPeekAtLastError());
	CUDA_ERROR(cudaMemcpy(rgba_data,dev_data,DIM*DIM*4*sizeof(GLubyte),cudaMemcpyDeviceToHost));

	t+=dt;
	if (t>25 || t<-25) dt=-dt;
}


static void idle_func(void)
{
	generateData();

	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, DIM*DIM*4,rgba_data,GL_DYNAMIC_DRAW_ARB);

	/*redisplay image*/
	glutPostRedisplay();

}

int main(int argc, char **argv)
{
    
/*for linux*/
#if defined(__linux__)
    setenv ("DISPLAY", ":0", 0);
#endif

  
	glutInit (&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGBA);
	glutInitWindowSize(DIM, DIM);
	glutCreateWindow("gl window");
	glewInit();

	glGenBuffers(1, &buffer_obj);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, buffer_obj);

	rgba_data = new GLubyte[DIM*DIM*4];
	CUDA_ERROR(cudaMalloc((void**)&dev_data,DIM*DIM*4*sizeof(GLubyte)));

	glutKeyboardFunc(key_func);
	glutDisplayFunc(draw_func);
	glutIdleFunc(idle_func);
	glutMainLoop();

	delete[] rgba_data;
	glDeleteBuffers(1,&buffer_obj);
	cudaFree(dev_data);

    return 0;    
}
