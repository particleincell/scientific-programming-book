/*Distributed Computing Lesson 8 

Example of using OpenGL to perform visualization 
of GPU computed results

step 1: demonstrates using openGL
*/

// OpenGL Graphics includes
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <iostream>

using namespace std;

//typedef unsigned int uint;
//typedef unsigned char uchar;

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

static void draw_func(void)
{
	glDrawPixels(DIM, DIM, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glutSwapBuffers();
}

/*some placeholder function to generate data*/
void generateData()
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
			if (z<1.0) {r=z*255;g=0;}
			else {r=255;g=(z-1)*255;}
			rgba_data[o+0] = r;
			rgba_data[o+1] = g;
			rgba_data[o+2] = 0;
			rgba_data[o+3] = 255;
		}
	t+=dt;
	if (t>25 || t<-25) dt=-dt;
}

/*updates data and causes redraw*/
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
	glutCreateWindow("Render 2");
	glewInit();

	glGenBuffers(1, &buffer_obj);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, buffer_obj);
	rgba_data = new GLubyte[DIM*DIM*4];

	glutKeyboardFunc(key_func);
	glutDisplayFunc(draw_func);
	//register idle function
	glutIdleFunc(idle_func);
	glutMainLoop();

	delete[] rgba_data;
	glDeleteBuffers(1,&buffer_obj);

    return 0;    
}
