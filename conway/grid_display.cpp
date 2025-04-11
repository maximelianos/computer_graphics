// Copyright 2020, Moscow State University,
// Author: Maksim Velikanov <maximelianos.m@gmail.com>.

#ifndef GRIDDISPLAY_H
#define GRIDDISPLAY_H

#include <glm/gtc/type_ptr.hpp>
#include "libjson/json/json.h"
#include "libjson/jsoncpp.cpp"

#include "common.h"

glm::mat4 MVP2() {
    // Projection matrix : 90 Field of View, WIDTH:HEIGHT ratio, clipping planes
    glm::mat4 projection = glm::ortho((float)-WIDTH / 2, (float)WIDTH / 2, (float)HEIGHT / 2, (float)-HEIGHT / 2, 1.0f, -2.0f);

    glm::mat4 view = glm::lookAt(
        env.camera_pos, // Camera position in World Space
        env.camera_pos + glm::vec3(0, 0, -1), // Looks at the origin
        glm::vec3(0, 1, 0)  // Camera rotation, "upward direction" (set to 0,-1,0 to look upside-down)
    );

    // Scale
    glm::mat4 mat_scale = glm::scale(glm::mat4(1.0f), glm::vec3(env.camera_scale, env.camera_scale, 1));
    //sc = glm::make_mat4(scale_mat_src);

    view = mat_scale * view;

    glm::mat4 model = glm::mat4(1.0f);
    //return model;
    return projection * view * model;
}



class GridDisplay {
public:
    // Game parameteres
    int game_mode = 0; // 0 - conway, 1 - generations, 2 - unicell, 3 - turmites
    int game_preset = 0;

    std::set<int> param_s{1, 2, 3}; // generations
    std::set<int> param_b{3, 4, 6, 8};
    int param_c = 9;
    int unicell_n = 17; // unicell
    int rand_seed = 1543;
    // turmites
    std::map<std::vector<int>, std::vector<int>> turm_rules;
    int turm_colors = 16;
    int turm_y, turm_x, turm_state, turm_dir; // dir: 0-up, 1-right, ...

    // Processing
    static const int N = 300;
    static const int triangles = N * N * 2 * 3 * 3; // Triangles per square, verticies per triangle, coors per vertex
    int *cell_field = nullptr;
    int *new_cell_field = nullptr;
    std::set<int> process_queue;
    std::set<int> new_process_queue;

    // Drawing
    float *triangle_pos = nullptr;
    float *triangle_color = nullptr;
    ShaderProgram *program = nullptr;
    GLuint g_vertexBufferObject = 0;
    GLuint g_vertexArrayObject = 0;
    GLuint color_buffer = 0;

    // State saving
    Json::Reader json_reader;
    Json::Value json_root;
    std::map<int, std::string> mode_name_dict = {
        {0, "conway"},
        {1, "sparks"},
        {2, "unicell"},
        {3, "turmites"}
    };


    GridDisplay() {
        // Shader programs
        std::unordered_map<GLenum, std::string> shaders;
        shaders[GL_VERTEX_SHADER]   = "grid_vertex.glsl";
    	shaders[GL_FRAGMENT_SHADER] = "grid_fragment.glsl";
        program = new ShaderProgram(shaders); GL_CHECK_ERRORS;

        // RAM arrays
        cell_field = new int[N * N];
        new_cell_field = new int[N * N];
        memset(cell_field, 0, N * N * sizeof(*cell_field));
        memset(new_cell_field, 0, N * N * sizeof(*new_cell_field));
        triangle_pos = new float[triangles];
        triangle_color = new float[triangles];

        // OpenGL arrays
        glGenBuffers(1, &g_vertexBufferObject); GL_CHECK_ERRORS;
        glGenVertexArrays(1, &g_vertexArrayObject); GL_CHECK_ERRORS;
        glGenBuffers(1, &color_buffer);

        env.camera_pos = glm::vec3(N / 2 * 10, N / 2 * 10, 0);

        // Load presets
        json_state_read();
        load_mode(game_mode);
    }

    inline int count_surround(int y, int x) {
        int sum = 0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dy != 0 || dx != 0) {
                    int y_new = (y + dy + N) % N;
                    int x_new = (x + dx + N) % N;
                    sum += cell_field[y_new * N + x_new];
                }
            }
        }
        return sum;
    }

    // "Generations" cell automata

    inline int dead_or_alive(int y, int x) { // Returns cell state: 0 - dead, 1 - alive, >1 - phase of dying (phases n+1, n, .., 2)
        int state = cell_field[y * N + x];
        if (state > 1) {
            if (state == 2) { // dying->dead
                return 0;
            } else {
                return state - 1;
            }
        } else {
            int sum = 0; // Count of alive neighbours
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dy != 0 || dx != 0) {
                        int y_new = (y + dy + N) % N;
                        int x_new = (x + dx + N) % N;
                        if (cell_field[y_new * N + x_new] == 1) {
                            sum += 1;
                        }
                    }
                }
            }

            if (state == 0) { // dead
                if (param_b.find(sum) != param_b.end()) {
                    return 1; // dead->alive
                } else {
                    return 0;
                }
            } else { // alive
                if (param_s.find(sum) != param_s.end()) {
                    return 1;
                } else {
                    return param_c; // alive->dying
                }
            }
        }
    }

    void generations_life_tick() {
        for (int y = 0; y < N; y++) {
            for (int x = 0; x < N; x++) {
                new_cell_field[y * N + x] = dead_or_alive(y, x);
            }
        }
        std::swap(cell_field, new_cell_field);
    }

    // "Conway"

    void conway_life_tick() {
        for (int y = 0; y < N; y++) {
            for (int x = 0; x < N; x++) {
                new_cell_field[y * N + x] = 0;
                int neighbours = count_surround(y, x);
                if (cell_field[y * N + x] == 0 && neighbours == 3) {
                    new_cell_field[y * N + x] = 1;
                } else if ( cell_field[y * N + x] == 1 && (neighbours == 2 || neighbours == 3) ) {
                    new_cell_field[y * N + x] = 1;
                }
            }
        }
        std::swap(cell_field, new_cell_field);
    }

    void conway_effective_life_tick() {
        for (int cell_n : process_queue) {
            int y = cell_n / N;
            int x = cell_n % N;

            new_cell_field[y * N + x] = 0;
            int neighbours = count_surround(y, x);
            if (cell_field[y * N + x] == 0) {
                if (neighbours == 3) { // 0->1, update
                    new_cell_field[y * N + x] = 1;

                    for (int dy = -1; dy <= 1; dy++) {
                        for (int dx = -1; dx <= 1; dx++) {
                            int new_y = (y + dy + N) % N;
                            int new_x = (x + dx + N) % N;
                            new_process_queue.insert(new_y * N + new_x);
                        }
                    }
                }
            } else if (cell_field[y * N + x] == 1) {
                if (neighbours == 2 || neighbours == 3) {
                    new_cell_field[y * N + x] = 1;
                } else { // 1->0, update
                    for (int dy = -1; dy <= 1; dy++) {
                        for (int dx = -1; dx <= 1; dx++) {
                            int new_y = (y + dy + N) % N;
                            int new_x = (x + dx + N) % N;
                            new_process_queue.insert(new_y * N + new_x);
                        }
                    }
                }
            }
        }
        std::swap(process_queue, new_process_queue);
        new_process_queue.clear();
        //std::swap(process_queue, new_process_queue);
        std::swap(cell_field, new_cell_field);
    }

    void alert_insert(int y, int x) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int new_y = (y + dy + N) % N;
                int new_x = (x + dx + N) % N;
                process_queue.insert(new_y * N + new_x);
            }
        }
    }

    // "Unicell"

    void fill_rand() {
        srand(rand_seed);
        for (int y = 0; y < N; y++) {
            for (int x = 0; x < N; x++) {
                int random = rand();
                random = random / ((double)RAND_MAX + 1) * (unicell_n);
                cell_field[y * N + x] = random;
            }
        }
    }

    void unicell_life_tick() {
        for (int y = 0; y < N; y++) {
            for (int x = 0; x < N; x++) {
                new_cell_field[y * N + x] = cell_field[y * N + x];
            }
        }
        for (int y = 0; y < N; y++) {
            for (int x = 0; x < N; x++) {
                int cur_cell = cell_field[y * N + x];
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        if (dy != 0 || dx != 0) {
                            int y_new = (y + dy + N) % N;
                            int x_new = (x + dx + N) % N;
                            if ((cur_cell - 1 + unicell_n) % unicell_n == cell_field[y_new * N + x_new]) {
                                new_cell_field[y_new * N + x_new] = cur_cell;
                            }
                        }
                    }
                }
            }
        }
        std::swap(cell_field, new_cell_field);
    }

    // "Turmites"

    void turm_move() { // With current pos and dir, move turmite to next cell
        if (turm_dir == 0) {
            turm_y--;
        } else if (turm_dir == 1) {
            turm_x++;
        } else if (turm_dir == 2) {
            turm_y++;
        } else if (turm_dir == 3) {
            turm_x--;
        }
        turm_y = (turm_y + N) % N; // Crossing the border
        turm_x = (turm_x + N) % N;
    }

    void turmite_life_tick() {
        int y = turm_y;
        int x = turm_x;
        std::vector<int> key{turm_state, cell_field[y * N + x]};
        std::vector<int> val = turm_rules[key];
        cell_field[y * N + x] = val[1];
        turm_state = val[0];
        turm_dir = (turm_dir + val[2] + 4) % 4;
        turm_move();
    }

    void general_life_tick() {
        if (game_mode == 0) {
            conway_life_tick();
        } else if (game_mode == 1) {
            generations_life_tick();
        } else if (game_mode == 2) {
            unicell_life_tick();
        } else if (game_mode == 3) {
            turmite_life_tick();
        }
    }

    // *********************************
    // Drawing functions
    // *********************************

    inline glm::vec3 cell_color(int state) {
        if (game_mode == 0) { // Conway
            return glm::vec3(state, state, state);
        } else if (game_mode == 1) { // Generations
            if (state == 0) {
                return glm::vec3(0, 0, 0);
            } else if (state == 1) {
                return glm::vec3(1, 1, 0);
            } else {
                return glm::vec3((float)state / param_c, 0, 0);
            }
        } else if (game_mode == 2) { // Unicell
            if (state <= 1 * unicell_n / 3) {
                float k = (state - 0 * (float)unicell_n / 3) / ((float)unicell_n / 3);
                return (1-k)*glm::vec3(1, 0, 0) + k*glm::vec3(0, 0, 1);
            } else if (state <= 2 * unicell_n / 3) {
                float k = (state - 1 * (float)unicell_n / 3) / ((float)unicell_n / 3);
                return (1-k)*glm::vec3(0, 0, 1) + k*glm::vec3(1, 0, 1);
            } else {
                float k = (state - 2 * (float)unicell_n / 3) / ((float)unicell_n / 3);
                return (1-k)*glm::vec3(1, 0, 1) + k*glm::vec3(1, 0, 0);
            }
            if (state == 0) {
                return glm::vec3(1, 1, 1);
            }
            return glm::vec3(0, 0, (float)state / unicell_n);
        } else { // Turmites
            switch(state) {
                case 0:
                    return glm::vec3(0, 0, 0);
                    break;
                case 1:
                    return glm::vec3(0, 0.5, 0);
                    break;
                case 2:
                    return glm::vec3(0, 1, 0);
                    break;
                case 3:
                    return glm::vec3(0.9, 0.9, 0.9);
                    break;
                case 4:
                    return glm::vec3(0.5, 0.0, 1.0);
                    break;
                case 5:
                    return glm::vec3(0.0, 1.0, 0.3);
                    break;
            }
            return glm::vec3((float)state / turm_colors, 0, 0);
        }
    }

    void init_draw() {
        int floats_in_square = 2 * 3 * 3; // Triangles per square, verticies per triangle, coors per vertex
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                int idx = (i * N + j) * floats_in_square; // Base idx
                int idx2 = 0; // Offset idx
                int di, dj;

                int di_list[] = {0, 1, 0, 0, 1, 1};
                int dj_list[] = {0, 1, 1, 0, 0, 1};
                for (int h = 0; h < 6; h++) {
                    di = di_list[h];
                    dj = dj_list[h];

                    glm::vec3 color(0, 0, 0);
                    if (cell_field[i * N + j]) {
                        color = glm::vec3(1, 0, 0);
                    }
                    color = cell_color(cell_field[i * N + j]);

                    if (dj == 0) { // x
                        triangle_pos[idx + idx2] = (j + dj) * (env.grid_cell_inner + 2 * env.grid_cell_offset) + env.grid_cell_offset;
                    } else {
                        triangle_pos[idx + idx2] = (j + dj) * (env.grid_cell_inner + 2 * env.grid_cell_offset) - env.grid_cell_offset;
                    }
                    triangle_color[idx + idx2] = color[0]; idx2++;

                    if (di == 0) { // y
                        triangle_pos[idx + idx2] = (i + di) * (env.grid_cell_inner + 2 * env.grid_cell_offset) + env.grid_cell_offset;
                    } else {
                        triangle_pos[idx + idx2] = (i + di) * (env.grid_cell_inner + 2 * env.grid_cell_offset) - env.grid_cell_offset;
                    }
                    //triangle_color[idx + idx2] = cell_color[i * N + j]; idx2++;
                    triangle_color[idx + idx2] = color[1]; idx2++;

                    triangle_pos[idx + idx2] = -1; // z
                    triangle_color[idx + idx2] = color[2]; idx2++;
                }
            }
        }

        // Create the buffer object for verticies' position
        //glGenBuffers(1, &g_vertexBufferObject); GL_CHECK_ERRORS;
        glBindBuffer(GL_ARRAY_BUFFER, g_vertexBufferObject); GL_CHECK_ERRORS;
        glBufferData(GL_ARRAY_BUFFER, triangles * sizeof(GLfloat), (GLfloat *)triangle_pos, GL_STATIC_DRAW); GL_CHECK_ERRORS;

        //glGenBuffers(1, &color_buffer);
        glBindBuffer(GL_ARRAY_BUFFER, color_buffer); GL_CHECK_ERRORS;
        glBufferData(GL_ARRAY_BUFFER, triangles * sizeof(GLfloat), (GLfloat *)triangle_color, GL_STATIC_DRAW); GL_CHECK_ERRORS;
    }

    void draw() {
        program->StartUseShader();
        glBindVertexArray(g_vertexArrayObject); GL_CHECK_ERRORS;

        GLuint vertexLocation = 0;
        glBindBuffer(GL_ARRAY_BUFFER, g_vertexBufferObject); GL_CHECK_ERRORS;
        glEnableVertexAttribArray(vertexLocation); GL_CHECK_ERRORS;
        glVertexAttribPointer(
            vertexLocation, // Location identical to layout in the shader
            3,              // Components in the attribute (3 coordinates)
            GL_FLOAT,       // Type of components
            GL_FALSE,       // Normalization
            0,              // Stride: byte size of structure
            0               // Offset (in bytes) of attribute inside structure
        ); GL_CHECK_ERRORS;

        GLuint colorLocation = 1;
        glBindBuffer(GL_ARRAY_BUFFER, color_buffer); GL_CHECK_ERRORS;
        glEnableVertexAttribArray(colorLocation); GL_CHECK_ERRORS;
        glVertexAttribPointer(
            colorLocation, // Location identical to layout in the shader
            3,              // Components in the attribute (3 coordinates)
            GL_FLOAT,       // Type of components
            GL_FALSE,       // Normalization
            0,              // Stride: byte size of structure
            0               // Offset (in bytes) of attribute inside structure
        ); GL_CHECK_ERRORS;

        GLuint MatrixID = glGetUniformLocation(program->GetProgram(), "MVP"); GL_CHECK_ERRORS;
        glm::mat4 mvp = MVP2();
        glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &mvp[0][0]); GL_CHECK_ERRORS;

        // Draw the triangles !
    	glDrawArrays(GL_TRIANGLES, 0, N * N * 2 * 3);       GL_CHECK_ERRORS;  // The last parameter of glDrawArrays is equal to VertexShader invocations

        program->StopUseShader();
    }

    // *********************************
    // State saving
    // *********************************

    void json_state_read() {
        std::ifstream in_file("game_save.json");
        std::string json(
            (std::istreambuf_iterator<char>(in_file)),
            (std::istreambuf_iterator<char>()       )
        );
        bool parse_success = json_reader.parse(json, json_root, false);
        die(!parse_success, "Cannot load state presets from JSON\n");
    }

    void set_state(std::string cells) {
        int pos = 0;
        while (1) {
            std::size_t openbr = cells.find('(', pos);
            if (openbr == std::string::npos) {
                break;
            }
            openbr += 1;
            int comma = cells.find(',', openbr);
            int closebr = cells.find(')', comma) - 1;
            std::string val_l = cells.substr(openbr, comma - openbr);
            std::string val_r = cells.substr(comma + 1, closebr - comma);
            int y = std::stoi(val_l);
            int x = std::stoi(val_r);
            cell_field[y * N + x] = 1;

            pos = closebr;
        }

        if (game_mode == 1) {
            int marker = cells.find('*', pos);
            param_s.clear();
            param_b.clear();
            int s1 = marker;
            marker = cells.find('*', s1 + 1);
            int s2 = marker;
            for (int i = s1 + 1; i < s2; i++) {
                int digit = std::stoi(cells.substr(i, 1));
                param_s.insert(digit);
            }
            marker = cells.find('*', s2 + 1);
            int s3 = marker;
            for (int i = s2 + 1; i < s3; i++) {

                int digit = std::stoi(cells.substr(i, 1));
                param_b.insert(digit);
            }
            marker = cells.find('*', s3 + 1);
            int s4 = marker;
            param_c = std::stoi(cells.substr(s3 + 1, s4 - s3 - 1));
        }

        if (game_mode == 2) {
            int marker = cells.find('*', pos);
            int s1 = marker;
            marker = cells.find('*', s1 + 1);
            int s2 = marker;
            unicell_n = std::stoi(cells.substr(s1 + 1, s2 - s1 - 1));

            marker = cells.find('*', s2 + 1);
            int s3 = marker;
            rand_seed = std::stoi(cells.substr(s2 + 1, s3 - s2 - 1));
        }
    }

    void set_rules(std::string rules) {
        turm_rules.clear();
        int pos = 0;
        while (1) {
            std::size_t openbr = rules.find('(', pos);
            if (openbr == std::string::npos) {
                break;
            }
            openbr += 1;
            int c1 = rules.find(',', openbr);
            int a = std::stoi(rules.substr(openbr, c1 - openbr));
            c1 += 1;
            int c2 = rules.find(',', c1);
            int b = std::stoi(rules.substr(c1, c2 - c1));
            c2 += 1;
            int c3 = rules.find(',', c2);
            int c = std::stoi(rules.substr(c2, c3 - c2));
            c3 += 1;
            int c4 = rules.find(',', c3);
            int d = std::stoi(rules.substr(c3, c4 - c3));
            c4 += 1;
            int closebr = rules.find(')', c4);
            int e = std::stoi(rules.substr(c4, closebr - c4));

            std::vector<int> key{a, b};
            std::vector<int> val{e, c, d}; // This is correct
            turm_rules[key] = val;

            pos = closebr;
        }
    }

    void load_mode(int mode=-1) {
        // Stop simulation and clear field
        clear_field();
        // Change mode
        if (mode != -1) {
            game_mode = mode;
        }
        if (game_mode == 3) {
            turm_y = N / 2;
            turm_x = N / 2;
            turm_state = 0;
            turm_dir = 1;
        }
        // Fill field
        if (mode == -1) {
            game_preset = (game_preset + 1) % json_root[mode_name_dict[game_mode]].size();
        } else {
            game_preset = 0;
        }
        const Json::Value result_value = json_root[mode_name_dict[game_mode]][game_preset];
        const std::string preset = result_value.asString();
        printf("MODE=%d PRESET_NUM=%d\nSTATE=%s\n", game_mode, game_preset, preset.c_str());
        if (game_mode == 3) {
            set_rules(preset);
        } else if (game_mode == 2) {
            set_state(preset);
            fill_rand();
        } else {
            set_state(preset);
        }
    }

    void clear_field() {
        env.editing_mode = true;
        memset(cell_field, 0, N * N * sizeof(*cell_field));
        memset(new_cell_field, 0, N * N * sizeof(*new_cell_field));
        //env.camera_pos = glm::vec3(N / 2 * 10, N / 2 * 10, 0);
        //env.camera_scale = env.default_camera_scale;
        //env.camera_scale_target = env.camera_scale;
    }

    void print_mode() {
        for (int y = 0; y < N; y++) {
            for (int x = 0; x < N; x++) {
                if (cell_field[y * N + x] == 1) {
                    cout << "(" << y << ',' << x << ')';
                }
            }
        }
        cout << "\n";
    }
};

#endif
