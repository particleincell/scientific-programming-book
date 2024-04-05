/*Distributed Computing Lesson 8 

Example of using OpenGL to perform visualization 
of GPU computed results

step 1: demonstrates using openGL
*/

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <iostream>

using namespace std;

#define DIM 500
#define PI 3.14159265359 

//responds to keyboard press
static void key_func(unsigned char key, int x, int y)
{
	switch(key)
	{
	case 27: /*esc*/
		exit(0);
	}
}

//this function is called whenever the windows needs to be redrawn
static void draw_func(void)
{
	glDrawPixels(DIM, DIM, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glutSwapBuffers();
}

/*some placeholder function to generate data*/
void generateData(GLubyte *rgba_data)
{
	static float t = 0;
	static float dt = 0.1;
	for (int j=0;j<DIM;j++)
		for (int i=0;i<DIM;i++)
		{
			int o = (j*DIM+i)*4;
			float fi = i/(float)(DIM-1.0);	/*convert to [0,1]*/
			float fj = j/(float)(DIM-1.0);
			float val = cos(fi*2*PI+t)*cos(t*fj*PI);
			float z = (val+1);	/*convert values to [0,2]*/
			int r,g;
			if (z<1.0) {r=(int)(z*255);g=0;}
			else {r=255;g=(int)((z-1)*255);}
			rgba_data[o+0] = r;
			rgba_data[o+1] = g;
			rgba_data[o+2] = 0;
			rgba_data[o+3] = 255;
		}
	t+=dt;
	if (t>25 || t<-25) dt=-dt;
}

/*main*/
int main(int argc, char **argv)
{    
/*for linux*/
#if defined(__linux__)
    setenv ("DISPLAY", ":0", 0);
#endif
	
	/*initialize openGL, using double buffer and RGBA mode*/
	glutInit (&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGBA);
	glutInitWindowSize(DIM, DIM);
	glutCreateWindow("Render 1");
	glewInit();	//needed to prevent crash in glGenBuffers

	//generate buffer object and bind it to pixel data
	GLuint buffer_obj;
	glGenBuffers(1, &buffer_obj);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, buffer_obj);

	//allocate memory to hold pixel data, each pixel is 4 bytes (RGBA)
	GLubyte *rgba_data = new GLubyte[DIM*DIM*4];
	//call our placeholder function to generate data
	generateData(rgba_data);

	//copy our data to the openGL buffer on the GPU
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, DIM*DIM*4,rgba_data,GL_DYNAMIC_DRAW_ARB);

	//add hooks for processing keyboard and draw requests
	glutKeyboardFunc(key_func);
	glutDisplayFunc(draw_func);

	//start main loop, leaves the window open and fires keyboard events
	glutMainLoop();

	//clean up
	delete[] rgba_data;
	glDeleteBuffers(1,&buffer_obj);

    return 0;    
}
