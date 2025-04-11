// Copyright 2020, Moscow State University,
// Author: Maksim Velikanov <maximelianos.m@gmail.com>.

#ifndef COMMON_H
#define COMMON_H

// C++ std libraries
#include <set>
#include <map>
#include <vector>
#include <random>
#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <stdlib.h>

using std::cout;

#include <glad/glad.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
using namespace glm;

// BMP reading
#include "bitmap_image.hpp"

//полезный макрос для проверки ошибок
//в строчке, где он был записан вызывает ThrowExceptionOnGLError, которая при возникновении ошибки opengl
//пишет в консоль номер текущей строки и название исходного файла
//а также тип ошибки
#define GL_CHECK_ERRORS ThrowExceptionOnGLError(__LINE__,__FILE__);


#define PI 3.1415926535897932384626433832795f


static void ThrowExceptionOnGLError(int line, const char *file)
{

	static char errMsg[512];

	//вызывается функция glGetError, проверяющая не произошла ли ошибка
	//в каком-то вызове opengl и если произошла, то какой код у ошибки
	GLenum gl_error = glGetError();

	if (gl_error == GL_NO_ERROR)
		return;

	switch (gl_error)
	{
	case GL_INVALID_ENUM:
		std::cerr << "GL_INVALID_ENUM file " << file << " line " << line << std::endl;
		break;

	case GL_INVALID_VALUE:
		std::cerr << "GL_INVALID_VALUE file " << file << " line " << line << std::endl;
		break;

	case GL_INVALID_OPERATION:
		std::cerr << "GL_INVALID_OPERATION file " << file << " line " << line << std::endl;
		break;

	case GL_STACK_OVERFLOW:
		std::cerr << "GL_STACK_OVERFLOW file " << file << " line " << line << std::endl;
		break;

	case GL_STACK_UNDERFLOW:
		std::cerr << "GL_STACK_UNDERFLOW file " << file << " line " << line << std::endl;
		break;

	case GL_OUT_OF_MEMORY:
		std::cerr << "GL_OUT_OF_MEMORY file " << file << " line " << line << std::endl;
		break;

	case GL_NO_ERROR:
		break;

	default:
		std::cerr << "Unknown error @ file " << file << " line " << line << std::endl;
		break;
	}

	if (gl_error != GL_NO_ERROR)
		throw std::runtime_error(errMsg);
}



static const float EPS = 1e-6;
static const GLsizei WIDTH = 1366, HEIGHT = 768;//640*2, HEIGHT = 480*2; //размеры окна
//static const GLsizei WIDTH = 640, HEIGHT = 480; //размеры окна

const int WORK_GROUP_SIZE = 128;

struct pos {
    float x, y, z, w;
};

struct var_env { // Global variables of the simulation
    float bkgcol[3] = {0.1, 0.1, 0.15};

    const float cursor_shock_thr = 100;
    glm::vec2 cursor_pos = glm::vec2(0, 0);
    glm::vec2 cursor_dir = glm::vec2(0, 0); // Difference of cur and prev pos

    const double screen_angle_unit = 1000; // -> 1 radian

    // float phi = 0; // Horizontal
    // float psi = 0.233; // Vertical
    // glm::vec3 camera_pos = glm::vec3(1.787, 0.241, -0.945);
    float phi = 0; // Horizontal
    float psi = 0; // Vertical
    glm::vec3 camera_pos = glm::vec3(0, 0, 0);
    const float min_camera_scale = (float)HEIGHT / 300 / 10 * 0.9; // / N / square_length * margin
    const float max_camera_scale = 13;
    const float default_camera_scale = 2.5;
    float camera_scale = default_camera_scale;
    float camera_scale_target = camera_scale;

    // Grid measures
    float grid_cell_inner = 9.5f;
    float grid_cell_offset = 0.25f;


    const glm::vec3 origin_forward_step = glm::vec3(0, 0, 0.01);
    const glm::vec3 origin_right_step = glm::vec3(-0.01, 0, 0);
    glm::vec3 forward_step = origin_forward_step; // Temporary variable
    glm::vec3 right_step = origin_right_step; // Temporary variable

    double output_time = 0; // Last console output time
    const double output_interval = 2; // In seconds

    // User interface state
    bool editing_mode = true;
    double tick_duration = (double)1 / 4; // 1 second

    bool w_pressed = false;
    bool a_pressed = false;
    bool s_pressed = false;
    bool d_pressed = false;

    bool left_pressed = false;
    bool right_pressed = false;

    bool wireframe_water = false;

    // Disturbance
    float dt_x, dt_z, dt_h;
    double time_pressed;
    bool released = false;
};

extern var_env env;

class Image {
public:
	const int COLOR_DEPTH = 3;
	unsigned char *image = nullptr;
	int n_rows, n_cols;
	int n_colors = COLOR_DEPTH;

    Image() {}
    Image(int rows, int cols);
	void load(const char *filename);
	unsigned char& operator()(int row, int col, int color);
	void merge(Image &buf, int n_start, int n_end);

	void set(int row, int col, glm::vec3 color);
	void set(int idx, glm::vec3 color);
	glm::vec3 get(int row, int col);

	void blur();
	void save(const std::string &path);

    ~Image() {
		if (image != nullptr) {
			delete image;
		}
	}
};

class GridDisplay;

namespace env_water {
    extern GridDisplay *grid_display;
}

extern void die(const int condition, const char *msg, void (* const exit_function)()=nullptr);
extern double get_wall_time();
extern glm::mat4 MVP();
extern float clamp(float x, float min_value, float max_value);

#endif
