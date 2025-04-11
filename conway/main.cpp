// Copyright 2020, Moscow State University,
// Author: Maksim Velikanov <maximelianos.m@gmail.com>.

// Internal includes
#include "common.h"
#include "ShaderProgram.h"

#include "grid_display.cpp"

using namespace env_water;

// Easy screen handling
#define GLFW_DLL
#include <GLFW/glfw3.h>



// ***********************************************************
// ********************** Global variables *******************
// ***********************************************************

int initGL()
{
	int res = 0;
	//грузим функции opengl через glad
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize OpenGL context" << std::endl;
		return -1;
	}

	std::cout << "Vendor: "   << glGetString(GL_VENDOR) << std::endl;
	std::cout << "Renderer: " << glGetString(GL_RENDERER) << std::endl;
	std::cout << "Version: "  << glGetString(GL_VERSION) << std::endl;
	std::cout << "GLSL: "     << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;

	return 0;
}


// ***********************************************************
// ***************** Keyboard and mouse **********************
// ***********************************************************

// Detect keyboard presses
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    switch (key) {
        case GLFW_KEY_ESCAPE:
            if (action == GLFW_PRESS) {
                die(1, "pressed escape");
            }
            break;
        case GLFW_KEY_ENTER:
            if (action == GLFW_PRESS) {
                env.editing_mode = !env.editing_mode;
            }
            break;
        case GLFW_KEY_RIGHT:
            if (action == GLFW_PRESS) {
                grid_display->general_life_tick();
            }
            break;
        case GLFW_KEY_DOWN:
            if (action == GLFW_PRESS) {
                env.tick_duration *= 2;
            }
            break;
        case GLFW_KEY_UP:
            if (action == GLFW_PRESS) {
                env.tick_duration /= 2;
            }
            break;
        case GLFW_KEY_Q:
            if (action == GLFW_PRESS) {
                //env.wireframe_water = !env.wireframe_water;
            }
            break;
        case GLFW_KEY_W:
            if (action == GLFW_PRESS) {
                env.w_pressed = true;
            } else if (action == GLFW_RELEASE) {
                env.w_pressed = false;
            }
            break;
        case GLFW_KEY_A:
            if (action == GLFW_PRESS) {
                env.a_pressed = true;
            } else if (action == GLFW_RELEASE) {
                env.a_pressed = false;
            }
            break;
        case GLFW_KEY_S:
            if (action == GLFW_PRESS) {
                env.s_pressed = true;
            } else if (action == GLFW_RELEASE) {
                env.s_pressed = false;
            }
            break;
        case GLFW_KEY_D:
            if (action == GLFW_PRESS) {
                env.d_pressed = true;
            } else if (action == GLFW_RELEASE) {
                env.d_pressed = false;
            }
            break;
        case GLFW_KEY_C:
            if (action == GLFW_PRESS) {
                grid_display->clear_field();
            }
            break;
        case GLFW_KEY_F1: // Change modes
            if (action == GLFW_PRESS) {
                grid_display->load_mode(0);
            }
            break;
        case GLFW_KEY_F2:
            if (action == GLFW_PRESS) {
                grid_display->load_mode(1);
            }
            break;
        case GLFW_KEY_F3:
            if (action == GLFW_PRESS) {
                grid_display->load_mode(2);
            }
            break;
        case GLFW_KEY_F4:
            if (action == GLFW_PRESS) {
                grid_display->load_mode(3);
            }
            break;
        case GLFW_KEY_N:
            if (action == GLFW_PRESS) {
                grid_display->load_mode(-1);
            }
            break;
        case GLFW_KEY_M:
            if (action == GLFW_PRESS) {
                grid_display->print_mode();
            }
            break;
    }
}

// Detect mouse movements
void cursor_position_callback(GLFWwindow *window, double xpos, double ypos) {
    env.cursor_dir = glm::vec2(xpos, ypos) - env.cursor_pos;
    env.cursor_pos = glm::vec2(xpos, ypos);
    if (env.cursor_dir[0] < env.cursor_shock_thr && env.cursor_dir[1] < env.cursor_shock_thr) {
        env.phi -= env.cursor_dir[0] / env.screen_angle_unit;
        env.psi += env.cursor_dir[1] / env.screen_angle_unit;
        env.psi = clamp(env.psi, -M_PI / 2 + 0.01, M_PI / 2 - 0.01);
    }
    if (env.left_pressed) { // Drag grid around
        float new_x = env.camera_pos.x - env.cursor_dir.x / env.camera_scale;
        float new_y = env.camera_pos.y - env.cursor_dir.y / env.camera_scale;
        if (0 < new_x && new_x < grid_display->N * (env.grid_cell_inner + 2 * env.grid_cell_offset)) {
            env.camera_pos.x = new_x;
        }
        if (0 < new_y && new_y < grid_display->N * (env.grid_cell_inner + 2 * env.grid_cell_offset)) {
            env.camera_pos.y = new_y;
        }
    }
    env.cursor_dir = glm::vec2(0, 0);
}

// Mouse press
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            env.left_pressed = true;
        } else {
            env.left_pressed = false;
        }
    }
    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        if (action == GLFW_PRESS) {
            env.right_pressed = true;
        } else {
            env.right_pressed = false;
        }
    }
}

// Mouse scroll
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    float new_scale;
    if (yoffset > 0) {
        new_scale = env.camera_scale_target * 1.2;
    } else if (yoffset < 0) {
        new_scale = env.camera_scale_target / 1.2;
    }
    if (env.min_camera_scale < new_scale && new_scale < env.max_camera_scale) {
        env.camera_scale_target = new_scale;
    }
}


// ***********************************************************
// ***** Camera movement, Model View Projection matrix *******
// ***********************************************************

/*glm::vec3 rotate_z(glm::vec3 v, float angle) {
    glm::vec3 rotation_axis = glm::vec3(0, 0, 1);
    glm::mat4 rotate_m = glm::rotate(angle, rotation_axis);
    return glm::mat3(rotate_m) * v;
}*/

void frame_key_update() {
    // Right-click with position
    if (env.right_pressed && env.editing_mode) { // Editing mode
        glm::vec2 mouse = env.cursor_pos;
        mouse.x -= WIDTH / 2;
        mouse.y -= HEIGHT / 2;
        mouse.x /= env.camera_scale;
        mouse.y /= env.camera_scale;
        mouse.x += env.camera_pos.x;
        mouse.y += env.camera_pos.y;
        //printf("V_X=%lf V_Y=%lf\n", mouse.x, mouse.y);
        int cell_x = mouse.x / (env.grid_cell_inner + 2 * env.grid_cell_offset);
        int cell_y = mouse.y / (env.grid_cell_inner + 2 * env.grid_cell_offset);
        //printf("CELL_X=%d CELL_Y=%d\n", cell_x, cell_y);
        grid_display->cell_field[cell_y * grid_display->N + cell_x] = !grid_display->cell_field[cell_y * grid_display->N + cell_x];
        grid_display->alert_insert(cell_y, cell_x);
        env.right_pressed = false;
    }
}

int main(int argc, char** argv)
{
    // GLFW and openGL initialisation
    die(!glfwInit(), "!glfwInit()");

    glfwWindowHint(GLFW_SAMPLES, 32); // Antialiasing
	//запрашиваем контекст opengl версии 3.3
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

    //GLFWwindow *window = glfwCreateWindow(WIDTH, HEIGHT, "OpenGL basic sample", nullptr, nullptr);
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "My Automata", glfwGetPrimaryMonitor(), nullptr);
    die(window == nullptr, "Failed to create GLFW window", glfwTerminate);

	glfwMakeContextCurrent(window);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

    die(initGL() != 0, "initGl() != 0");

    //Reset any OpenGL errors which could be present for some reason
	GLenum gl_error = glGetError();
	while (gl_error != GL_NO_ERROR) {
		gl_error = glGetError();
    }



	//создание шейдерной программы из двух файлов с исходниками шейдеров
	//используется класс-обертка ShaderProgram
	std::unordered_map<GLenum, std::string> shaders;

    //glfwSwapInterval(0); // force 60 frames per second

    //Создаем и загружаем геометрию поверхности
    GLuint g_vertexBufferObject = 0;
    GLuint g_vertexArrayObject = 0;
    GLuint color_buffer = 0;

    // VERY IMPORTANT
    //water_surface = new WaterSurface();
    //TexturedWall floor(FLOOR0);
    //common = new TexturedWall(FLOOR0);
    // floor0 = new TexturedWall(FLOOR0);
    // wall1 = new TexturedWall(WALL1);
    // wall2 = new TexturedWall(WALL2);
    // wall3 = new TexturedWall(WALL3);
    // wall4 = new TexturedWall(WALL4);
    // ball5 = new TexturedSphere(BALL5);
    grid_display = new GridDisplay();


	//цикл обработки сообщений и отрисовки сцены каждый кадр
    double last_refresh = get_wall_time();
    const double frame_interval = 1.0 / 40.0;

    glfwSetKeyCallback(window, key_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetScrollCallback(window, scroll_callback);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    //glDisable(GL_CULL_FACE);
    //glEnable(GL_BLEND);
    //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    //texture_sq.g_texture = floor.rendered_texture;

    double last_tick_time = get_wall_time();

    int frames = 0;
    float render_time = 0;
	while (!glfwWindowShouldClose(window))
	{
        last_refresh = get_wall_time();

        // Screen control elements
        for (int i = 0; i < 30; i++) {
		    glfwPollEvents();
        }
        float scale_speed = 0.15;
        env.camera_scale = env.camera_scale * (1 - scale_speed) + env.camera_scale_target * scale_speed;
        if (env.camera_scale < 0.65) {
            env.grid_cell_offset = 0;
        } else {
            env.grid_cell_offset = 0.25;
        }
        env.grid_cell_inner = 10 - 2 * env.grid_cell_offset;

        if (!env.editing_mode) {
            double cur_time = get_wall_time();
            while (cur_time - last_tick_time > env.tick_duration) {
                last_tick_time += env.tick_duration;
                grid_display->general_life_tick();
            }
        } else {
            last_tick_time = get_wall_time();
        }

        frame_key_update();

        // очистка и заполнение экрана цветом
        glViewport  (0, 0, WIDTH, HEIGHT); GL_CHECK_ERRORS;
        glClearColor(env.bkgcol[0], env.bkgcol[1], env.bkgcol[2], 0.0f); GL_CHECK_ERRORS;
        glClear     (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT); GL_CHECK_ERRORS;

        grid_display->init_draw();
        grid_display->draw();

        // Sleeping (constant frame rate), output
        frames++;
        double cur_time = get_wall_time();
        render_time += cur_time - last_refresh;
        //printf("Total render time %lf\n", cur_time - last_refresh);

        double cur_interval = cur_time - env.output_time;
        if (cur_interval > env.output_interval) {
            env.output_time = cur_time;
            // printf("TIME %lf CAMERA POS %lf %lf %lf\n", cur_time, env.camera_pos[0], env.camera_pos[1], env.camera_pos[2]);
            // printf("SCALE = %lf POS = %lf,%lf\n", env.camera_scale, env.camera_pos.x, env.camera_pos.y);
            // printf("FRAME INTERVAL %lf\n", frame_interval);
            // //printf("RENDER TIME %lf\n", cur_time - last_refresh);
            // printf("RENDER TIME %lf\n", render_time / frames);
            // printf("\n");
            render_time = 0;
            frames = 0;
        }

        cur_interval = cur_time - last_refresh;
        //printf("Sleep for %lf\n", (frame_interval - cur_interval));
        if (cur_interval < frame_interval) { // Sleep until beginning of next frame
            usleep((frame_interval - cur_interval) * 1000000);
        } else {
            printf("Negative sleep: %lf\n", frame_interval - cur_interval);
        }

        glfwSwapBuffers(window);
	}

	//очищаем vboи vao перед закрытием программ
	//glDeleteVertexArrays(1, &g_vertexArrayObject);
    //glDeleteBuffers(1,      &g_vertexBufferObject);
    //water_surface->free();

	glfwTerminate();

	return 0;
}
