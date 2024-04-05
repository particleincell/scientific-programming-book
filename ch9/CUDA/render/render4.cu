/*Distributed Computing Lesson 8 

Example of using OpenGL to perform visualization 
of GPU computed results

step 3: uses CUDA to generate and copy to GL
*/

// OpenGL Graphics includes
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "device_launch_parameters.h"

using namespace std;

#define CUDA_ERROR(f) {if (f!=cudaSuccess) {cerr<<cudaGetErrorString(f)<<" on line "<<__LINE__<<endl;exit(-1);}}

#define DIM 500
#define PI 3.14159265359 

/*globals*/
GLuint buffer_obj;
GLubyte *dev_data;
cudaGraphicsResource *resource;

static void key_func(unsigned char key, int x, int y)
{
	switch(key)
	{
	case 27: /*esc*/
		//CUDA_ERROR(cudaGraphicsUnregisterResource(resource));
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB,0);
		glDeleteBuffers(1,&buffer_obj);
		exit(0);
	}
}

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

void generateData()
{
	static float t = 0;
	static float dt = 0.1;

	size_t size;
	//map openGL graphics resource and get device pointer to it
	CUDA_ERROR(cudaGraphicsMapResources(1,&resource,NULL));
	CUDA_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&dev_data,&size,resource));

	dim3 dims = {DIM,DIM,1};
	gen_data_gpu<<<dims,1>>>(dev_data,t);
	CUDA_ERROR(cudaPeekAtLastError());
	
	//release resource to openGL
	CUDA_ERROR(cudaGraphicsUnmapResources(1,&resource,NULL));

	t+=dt;
	if (t>25 || t<-25) dt=-dt;
}


static void idle_func(void)
{
	generateData();

	/*redisplay image*/
	glutPostRedisplay();

}

int main(int argc, char **argv)
{
    
/*for linux*/
#if defined(__linux__)
    setenv ("DISPLAY", ":0", 0);
#endif

	/*get device id*/
	cudaDeviceProp prop;
	int dev;
	memset(&prop,0,sizeof(cudaDeviceProp));
	prop.major = 1;
	prop.minor = 1;
	CUDA_ERROR(cudaChooseDevice(&dev,&prop));
	cout<<"Device id is "<<dev<<endl;
	CUDA_ERROR(cudaGLSetGLDevice(dev));

	glutInit (&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGBA);
	glutInitWindowSize(DIM, DIM);
	glutCreateWindow("Render 4");
	glewInit();

	glGenBuffers(1, &buffer_obj);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, buffer_obj);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, DIM*DIM*4, NULL, GL_DYNAMIC_DRAW_ARB);

	CUDA_ERROR(cudaPeekAtLastError());
	CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&resource,buffer_obj,cudaGraphicsMapFlagsNone));
	
	glutKeyboardFunc(key_func);
	glutDisplayFunc(draw_func);
	glutIdleFunc(idle_func);
	glutMainLoop();

	glDeleteBuffers(1,&buffer_obj);

    return 0;    
}
