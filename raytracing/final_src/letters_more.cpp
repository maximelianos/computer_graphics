#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include <thread>
#include <string>

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

#include "bitmap_image.hpp"

using std::cout;
using std::vector;
using std::shared_ptr;
using std::make_shared;
using std::thread;

float EPS = 1e-5;
const float INF = 1e6;
const float PI = 3.14159265358979323846;
const float PI2 = 6.283185307179586232;



// ******************************************************************* //
// ************************ General functions ************************ //
// ******************************************************************* //

template<typename Tnum>
Tnum inline abs(Tnum n) {
    if (n < 0) {
		return -n;
	}
    return n;
}

template<typename Tnum1, typename Tnum2>
bool inline eq(Tnum1 a, Tnum2 b) {
    return abs(a - b) < EPS;
}

float RAND_MAX_ONE = RAND_MAX + 1.0;
unsigned counter = 0;

inline float random_double() {
	return (float) (random() / RAND_MAX_ONE);
}

void die(int condition, const char *msg) {
	if (condition) {
		printf("Exiting: %s\n", msg);
		exit(0);
	}
}



class vec;
float dot(const vec&, const vec&);
vec cross(const vec&, const vec&);

class vec {
public:
    float v[3];

    vec() {}

    vec(float x, float y, float z) {
		v[0] = x;
		v[1] = y;
		v[2] = z;
	}
	
	float inline x() const { return v[0]; }
	
	float inline y() const { return v[1]; }
	
	float inline z() const { return v[2]; }
	
	float inline operator[](int idx) const { return v[idx]; }
	
	float inline & operator[](int idx) { return v[idx]; }

    float inline len2() const {
        return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    }

    float inline len() const {
        return sqrt(len2());
    }

    vec inline operator+(const vec &u) const {
        return vec(v[0] + u[0], v[1] + u[1], v[2] + u[2]);
    }
    
    vec inline operator+=(const vec &u) {
		v[0] += u[0];
		v[1] += u[1];
		v[2] += u[2];
		return *this;
	}

    vec inline operator-(const vec &u) const {
        return vec(v[0] - u[0], v[1] - u[1], v[2] - u[2]);
    }

    vec inline operator-() const {
        return vec(-v[0], -v[1], -v[2]);
    }

    vec inline operator*(float k) const {
        return vec(v[0] * k, v[1] * k, v[2] * k);
    }
    
    vec inline operator*(const vec &u) const {
        return vec(v[0] * u[0], v[1] * u[1], v[2] * u[2]);
    }
    
    vec inline operator/(float k) const {
        return vec(v[0] / k, v[1] / k, v[2] / k);
    }
    
    friend inline vec operator*(float t, const vec &v);

    vec inline unit() const {
        float d = len();
        return vec(v[0] / d, v[1] / d, v[2] / d);
    }
    
    vec inline clamp(float min_val, float max_val) const {
		return vec(fmax(min_val, fmin(max_val, v[0])),
				fmax(min_val, fmin(max_val, v[1])),
				fmax(min_val, fmin(max_val, v[2]))
		);
	}

    bool inline operator==(const vec &u) const {
        return eq(v[0], u[0]) && eq(v[1], u[1]) && eq(v[2], u[2]);
    }

    bool inline operator!=(const vec &u) const {
        return !(*this == u);
    }

    bool inline operator||(const vec &u) const { // Parallel
        return eq(cross(*this, u).len2(), 0);
    }
};

std::istream& operator>>(std::istream &in, vec &a) {
    in >> a[0] >> a[1] >> a[2];
    return in;
}

std::ostream& operator<<(std::ostream &out, const vec &a) {
    out << a[0] << " " << a[1] << " " << a[2];
    return out;
}

vec inline operator*(float k, const vec &v) {
    return vec(k * v[0], k * v[1], k * v[2]);
}

float inline dot(const vec &a, const vec &b) { // (a, b) = x1 * x2 + y1 * y2
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

vec inline cross(const vec &a, const vec &b) { // [a, b] = x1 * y2 - y1 * x2
    return vec(a[1] * b[2] - a[2] * b[1], -a[0] * b[2] + a[2] * b[0], a[0] * b[1] - a[1] * b[0]);
}

float uni(float angle) { // Making angles universal - from 0 to 2pi
    while (angle < 0) {
        angle += PI2;
    }
    while (angle >= PI2) {
        angle -= PI2;
    }
    return angle;
}

float inline rad(float degrees) {
	return degrees * M_PI / 180;
}

// ******************************************************************* //
// ***************************** Image saving ************************ //
// ******************************************************************* //


class Image {
public:
	const int COLOR_DEPTH = 3;
	unsigned char *image = nullptr;
	int n_rows, n_cols;
	int n_colors = COLOR_DEPTH;
	
	Image() {}
	
	Image(int rows, int cols) {
		image = new unsigned char[rows * cols * COLOR_DEPTH];
		n_rows = rows;
		n_cols = cols;
	}
	
	void load(const char *filename) {
		bitmap_image img(filename);
		n_rows = img.height();
		n_cols = img.width();
		image = new unsigned char[n_rows * n_cols * COLOR_DEPTH];
		for (int y = 0; y < n_rows; ++y) {
			for (int x = 0; x < n_cols; ++x) {
				rgb_t colour;

				img.get_pixel(x, y, colour);
				int idx = y * (n_cols * n_colors) + x * n_colors;
				image[idx] = colour.red;
				image[idx + 1] = colour.green;
				image[idx + 2] = colour.blue;
			}
		}
	}

	unsigned char& operator()(int row, int col, int color) {
		return image[row * (n_cols * n_colors) + col * n_colors + color];
	}
	
	void merge(Image &buf, int n_start, int n_end) {
		for (int row = n_start; row < n_end; row++) {
			for (int col = 0; col < n_cols; col++) {
				int idx = row * (n_cols * n_colors) + col * n_colors;
				image[idx] = buf.image[idx];
				image[idx + 1] = buf.image[idx + 1];
				image[idx + 2] = buf.image[idx + 2];
			}
		}
	}
	
	void set(int row, int col, vec color) {
		int idx = row * (n_cols * n_colors) + col * n_colors;
		image[idx] = int(color[0]);
		image[idx + 1] = int(color[1]);
		image[idx + 2] = int(color[2]);
	}
	
	void set(int idx, vec color) {
		image[idx] = color[0];
		image[idx + 1] = color[1];
		image[idx + 2] = color[2];
	}
	
	vec get(int row, int col) {
		int idx = row * (n_cols * n_colors) + col * n_colors;
		return vec(image[idx], image[idx + 1], image[idx + 2]);
	}
	
	void blur() {
		float kernel[25] = {};
		float sigma = 1;
		for (int y = -2; y < 3; y++) {
			for (int x = -2; x < 3; x++) {
				float g = exp(-(x * x + y * y) / (2 * sigma * sigma));
				g = g / (2 * M_PI * sigma * sigma);
				kernel[(y + 2) * 5 + (x + 2)] = g;
			}
		}
		unsigned char *buf = new unsigned char[n_rows * n_cols * COLOR_DEPTH];
		for (int i = 2; i < n_rows - 2; i++) {
			for (int j = 2; j < n_cols - 2; j++) {
				for (int c = 0; c < 3; c++) {
					float tmp = 0;
					for (int y = -2; y < 3; y++) {
						for (int x = -2; x < 3; x++) {
							tmp += kernel[(y + 2) * 5 + (x + 2)] * image[(i + y) * (n_cols * n_colors) + (j + x) * n_colors + c];
						}
					}
					buf[i * (n_cols * n_colors) + j * n_colors + c] = tmp;
				}
			}
		}
		delete image;
		image = buf;
	}

	void save(const std::string &path) {
		blur();
		
		bitmap_image bitmapbuf(n_cols, n_rows);
		bitmapbuf.clear();
		int idx = 0;
		for (int y = 0; y < n_rows; y++) {
			for (int x = 0; x < n_cols; x++) {
				rgb_t c;
				c.red = image[idx];
				c.green = image[idx + 1];
				c.blue = image[idx + 2];
				bitmapbuf.set_pixel(x, y, c);

				idx += n_colors;
			}
		}
		bitmapbuf.save_image(path);
		
#if 0
		FILE *fout = fopen(path.c_str(), "w");
		fprintf(fout, "P3\n%d %d\n255\n", n_cols, n_rows);
		int idx = 0;
		for (int y = n_rows - 1; y >= 0; y--) {
			for (int x = 0; x < n_cols; x++) {
				fprintf(fout, "%d %d %d\n",
						image[idx],
						image[idx + 1],
						image[idx + 2]);
				idx += n_colors;
			}
		}
		fclose(fout);
#endif
	}

	~Image() {
		if (image != nullptr) {
			delete image;
		}
	}
};


// ******************************************************************* //
// *************************** Basic geometry ************************ //
// ******************************************************************* //


vec mult_v_mat(vec v, float *mat) {
	vec res;
	res[0] = mat[0 * 3 + 0] * v[0] + mat[0 * 3 + 1] * v[1] + mat[0 * 3 + 2] * v[2];
	res[1] = mat[1 * 3 + 0] * v[0] + mat[1 * 3 + 1] * v[1] + mat[1 * 3 + 2] * v[2];
	res[2] = mat[2 * 3 + 0] * v[0] + mat[2 * 3 + 1] * v[1] + mat[2 * 3 + 2] * v[2];
	return res;
}

void mult_mat(float *A, float *B) {
	// A = mult_mat(A, B)
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			float tmp = 0;
			for (int k = 0; k < 3; k++) {
				tmp += A[i * 3 + k] * B[k * 3 + j];
			}
			A[i * 3 + j] = tmp;
		}
	}
}

void rot_x(float theta, float *mat) {
	// mat is a return parameter
	mat[0 * 3 + 0] = 1;
	mat[0 * 3 + 1] = 0;
	mat[0 * 3 + 2] = 0;
	mat[1 * 3 + 0] = 0;
	mat[1 * 3 + 1] = cos(theta);
	mat[1 * 3 + 2] = -sin(theta);
	mat[2 * 3 + 0] = 0;
	mat[2 * 3 + 1] = sin(theta);
	mat[2 * 3 + 2] = cos(theta);
}

void rot_y(float theta, float *mat) {
	// mat is a return parameter
	mat[0 * 3 + 0] = cos(theta);
	mat[0 * 3 + 1] = 0;
	mat[0 * 3 + 2] = sin(theta);
	mat[1 * 3 + 0] = 0;
	mat[1 * 3 + 1] = 1;
	mat[1 * 3 + 2] = 0;
	mat[2 * 3 + 0] = -sin(theta);
	mat[2 * 3 + 1] = 0;
	mat[2 * 3 + 2] = cos(theta);
}

void rot_z(float theta, float *mat) {
	// mat is a return parameter
	mat[0 * 3 + 0] = cos(theta);
	mat[0 * 3 + 1] = -sin(theta);
	mat[0 * 3 + 2] = 0;
	mat[1 * 3 + 0] = sin(theta);
	mat[1 * 3 + 1] = cos(theta);
	mat[1 * 3 + 2] = 0;
	mat[2 * 3 + 0] = 0;
	mat[2 * 3 + 1] = 0;
	mat[2 * 3 + 2] = 1;
}



class Ray {
	vec A;
	vec B;
	float th;

public:
	Ray() {}
	Ray(const vec &a, const vec &b, const float theta) : A(a), B(b), th(theta) {}
	vec origin() const { return A; }
	vec direction() const { return B; }
	float theta() const { return th; }   
	vec point_at_parameter(float t) const { return A + t*B; }
};



class Camera {
	vec origin;
    float h_fov, v_fov;
	vec lower_left_corner;
	vec horizontal;
	vec vertical;

public:
	Camera(float horizontal_rad_fov=M_PI, float vertical_rad_fov=M_PI, float rx=0, vec pos=vec(0, 0, 0)) {
		float mat_x[9];
		float mat_y[9];
		float mat_z[9];
		float phi;
		
		rot_x(rx * M_PI / 180, mat_x);
		rot_y(0 * M_PI / 180, mat_y);
		rot_z(0 * M_PI / 180, mat_z);
		mult_mat(mat_x, mat_y);
		mult_mat(mat_x, mat_z);
		
		float half_h = tan(horizontal_rad_fov / 2);
		float half_v = tan(vertical_rad_fov / 2);
		lower_left_corner = mult_v_mat(vec(-half_h, -half_v, -1.0), mat_x);
		horizontal = vec(half_h * 2, 0.0, 0.0);
		vertical = mult_v_mat(vec(0.0, half_v * 2, 0.0), mat_x);
		origin = pos;
	}
	Ray get_ray(float u, float v) const {
		return Ray(origin, lower_left_corner + u*horizontal + v*vertical, 1.0);
	}


};


struct material;

struct hit_record {
    float t;
    vec p;
    vec normal;
    
    float theta = 1.0;
    
    vec ambient_color = vec(0, 0, 0);
    vec force_color = vec(-1, 0, 0);
	
	bool pass_arg = false;
    material *mat_ptr;
};

float inline ffmin(float a, float b) { return a <= b ? a : b; }
float inline ffmax(float a, float b) { return a >= b ? a : b; }

vec inline vec_min(const vec &a, const vec &b) {
	return vec(ffmin(a[0], b[0]), ffmin(a[1], b[1]), ffmin(a[2], b[2]));
}

vec inline vec_max(const vec &a, const vec &b) {
	return vec(ffmax(a[0], b[0]), ffmax(a[1], b[1]), ffmax(a[2], b[2]));
}

struct slab {
	vec min;
	vec max;
	
	slab() {}
	slab(const vec &a, const vec &b) : min(a), max(b) {}

	bool inline hit(const Ray &ray, double tmin, double tmax) const {
		for (int a = 0; a < 3; a++) {
			float p0 = (min[a] - ray.origin()[a]) / ray.direction()[a];
			float p1 = (max[a] - ray.origin()[a]) / ray.direction()[a];
			float t0 = ffmin(p0, p1);
			float t1 = ffmax(p0, p1);
			tmin = ffmax(t0, tmin);
			tmax = ffmin(t1, tmax);
			if (tmax <= tmin) {
				return false;
			}
		}
		return true;
	}
};

std::ostream& operator<<(std::ostream &out, const slab &box) {
    out << box.min << " " << box.max;
    return out;
}

slab surround_slab(const slab &a, const slab &b) {
	return slab(vec_min(a.min, b.min), vec_max(a.max, b.max));
}



// Abstract class for rendering any surface or volume
class Hittable  {
public:
	virtual bool hit(
		const Ray &r, float t_min, float t_max, hit_record &rec) const = 0;
	virtual slab bound() const = 0;
};



bool compare_x(const shared_ptr<Hittable> &a, const shared_ptr<Hittable> &b) {
	return a->bound().min.x() < b->bound().min.x();
}

bool compare_y(const shared_ptr<Hittable> &a, const shared_ptr<Hittable> &b) {
	return a->bound().min.y() < b->bound().min.y();
}

bool compare_z(const shared_ptr<Hittable> &a, const shared_ptr<Hittable> &b) {
	return a->bound().min.z() < b->bound().min.z();
}



class BVH_node : public Hittable {
public:
	bool is_leaf = false;
	shared_ptr<Hittable> left;
	shared_ptr<Hittable> right;
	slab box;
	
	BVH_node() {}
	BVH_node(vector< shared_ptr<Hittable> > list) :
			BVH_node(list, 0, list.size()) {}
	BVH_node(vector< shared_ptr<Hittable> > &objects, int beg_idx, int end_idx, int depth=0) {
		bool cprint = false;
		int axis = counter;
		counter = (counter + 1) % 3;
		auto comparator = compare_z;
		if (axis == 0) {
			comparator = compare_x;
		} else if (axis == 1) {
			comparator = compare_y;
		}
		
		int len = end_idx - beg_idx;
		
		if (len == 1) {
			is_leaf = true;
			left = objects[beg_idx];
			box = left->bound();

			if (cprint) {
				for (int i = 0; i < depth; i++) {
					cout << " ";
				}
				cout << "LEAF\n";
			}
		} else {
			std::sort(objects.begin() + beg_idx, objects.begin() + end_idx, comparator);

			int mid = (beg_idx + end_idx) / 2;
			
			if (cprint) {
				for (int i = 0; i < depth; i++) {
					cout << " ";
				}
				cout << "LEFT:\n";
			}
			left = make_shared<BVH_node>(objects, beg_idx, mid, depth+1);
			if (cprint) {
				for (int i = 0; i < depth; i++) {
					cout << " ";
				}
				cout << "RIGHT:\n";
			}
			right = make_shared<BVH_node>(objects, mid, end_idx, depth+1);
			box = surround_slab(left->bound(), right->bound());
			if (cprint) {
				for (int i = 0; i < depth; i++) {
					cout << " ";
				}
				cout << "BOX: " << box << "\n";
			}
		}
		if (cprint && depth == 0) {
			cout << "\n\n\n\n";
		}
	}

	virtual bool hit(const Ray &r, float t_min, float t_max, hit_record &rec) const {
		if (!box.hit(r, t_min, t_max)) {
			return false;
		}

		bool hit_left = left->hit(r, t_min, t_max, rec);
		bool hit_right = false;
		if (!is_leaf) {
			if (hit_left) {
				hit_right = right->hit(r, t_min, rec.t, rec);
			} else {
				hit_right = right->hit(r, t_min, t_max, rec);
			}
		}

		return hit_left || hit_right;
	}
	
	virtual slab bound() const {
		return box;
	}
};



class HittableList: public Hittable {
public:
	vector< shared_ptr<Hittable> > l; // Without pointer it won't compile
	shared_ptr<BVH_node> tree_root;



	HittableList() {}
	HittableList(const vector< shared_ptr<Hittable> > &list) { l = list; }

    void add(const shared_ptr<Hittable> &object) {
        l.push_back(object);
    }
    
    void construct_BVH() {
		cout << "Creating BVH\n";
		tree_root = make_shared<BVH_node>(l);
		cout << "BVH created\n";
	}
	
	virtual bool hit(
		const Ray &r, float t_min, float t_max, hit_record& rec) const;
	
	slab bound() const {
		return slab(vec(0,0,0),vec(0,0,0));
	}
};

bool HittableList::hit(const Ray &r, float t_min, float t_max,
                        hit_record &rec) const {
	hit_record temp_rec;
    bool hit_anything = false;
    double closest_so_far = t_max;
    for (int i = 0; i < l.size(); i++) {
        if (l[i]->hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
    return tree_root->hit(r, t_min, t_max, rec);
}


enum {
	MAT_DIFFUSE, MAT_METAL, MAT_DIELECTRIC
};

struct material {
	int mat_type = MAT_DIFFUSE;
	// color = (diffuse + specular) * (1 - k_mirror) +
	// + (reflect * k_reflect + refract * (1 - k_reflect)) * k_mirror
	float k_diffuse = 0.5;
	
	float k_mirror = 1.0;
	
	float k_reflect = 0.1;
	
	float theta = 1;
	float theta_inv = 1;
	float theta_sq = 1;
	float theta_sq_inv = 1;
	
	vec ambient_color = vec(1, 1, 1);
	
	material(
		const int mat_type,
		const float theta=1.0) :
			mat_type(mat_type),
			theta(theta) {

		theta_inv = 1 / theta;
		theta_sq = theta * theta;
		theta_sq_inv = 1 / theta_sq;
	}
	
	material* clone() {
		return new material(*this);
	}
};

inline vec reflect(const vec &vec_in, const vec &normal) {
    return vec_in - 2 * normal * dot(vec_in, normal);
}



class Sphere: public Hittable {
public:
	vec C;
	float r;
	material *mat_ptr = nullptr;

	Sphere() {}
	Sphere(
        const vec &center,
        const float radius,
        material* const mat_ptr) :
			C(center),
			r(radius),
			mat_ptr(mat_ptr) {}

	vec center() const 	{ return C; }
	float radius() const 	{ return r; }
	~Sphere() { // Ugly implementation
		if (mat_ptr != nullptr) {
			delete mat_ptr;
		}
	}
	
	bool hit(const Ray &ray, float t_min, float t_max, hit_record &rec) const {
		// t^2 (B, B) + 2t (B, A-C) + (A-C, A-C) - R^2 >= 0 - discriminant
		float a = dot(ray.direction(), ray.direction());
		float b = dot(ray.direction(), ray.origin() - C); // Ommited *2
		float c = dot(ray.origin() - C, ray.origin() - C) - r * r;
		float d = b * b - a * c;

		if (d > 0) {
			float x1 = (-b - sqrt(d)) / a;
			if (t_min < x1 && x1 < t_max) {
				rec.t = x1;
				rec.p = ray.point_at_parameter(x1);
				rec.normal = (rec.p - C) / r;
                rec.mat_ptr = mat_ptr;
                rec.ambient_color = mat_ptr->ambient_color;
				return true;
			}

			float x2 = (-b + sqrt(d)) / a;
			if (t_min < x2 && x2 < t_max) {
				rec.t = x2;
				rec.p = ray.point_at_parameter(x2);
				rec.normal = (rec.p - C) / r;
                rec.mat_ptr = mat_ptr;
                rec.ambient_color = mat_ptr->ambient_color;
				return true;
			}
		}

		return false;
	}
	
	slab bound() const {
		return slab(C - vec(r, r, r), C + vec(r, r, r));
	}
};

class InfinitePlane: public Hittable {
	vec p; // Origin point
	vec u, v; // Direction vectors, counter-clockwise
	vec normal;

	float a, b, c, d;
    material *mat_ptr = nullptr;

public:
	InfinitePlane() {}
	~InfinitePlane() {
		if (mat_ptr != nullptr) {
			delete mat_ptr;
		}
	}
	InfinitePlane(
        const vec &point, const vec &dir1, const vec &dir2, material *mat_ptr) :
			mat_ptr(mat_ptr) {

		p = point;
		u = dir1;
		v = dir2;

		float tx = v.z() * u.y() - v.y() * u.z();
		a = -u.y() * tx;
		float ty = u.x() * v.z() - u.z() * v.x();
		b = u.y() * ty;
		float tz = v.x() * u.y() - v.y() * u.x();
		c = u.y() * tz;
		d = (p.x() * u.y() - p.y() * u.x()) * tx -
		    (p.z() * u.y() - p.y() * u.z()) * tz;

		normal = (vec(tx, -ty, -tz)).unit();

		v = cross(normal, u);
	}

	bool hit(const Ray &ray, float t_min, float t_max, hit_record &rec) const {
		// M = p + t_1*u + t_2*v = q + t_3*s =>
		// t_1*u + t_2*v - t_3*s = q - p
		vec s = -ray.direction();
		vec origin = ray.origin() - p;
		float mat[3][4] = {
			{u.x(), v.x(), s.x(), origin.x()},
			{u.y(), v.y(), s.y(), origin.y()},
			{u.z(), v.z(), s.z(), origin.z()}
		};
		float det_3 = mat[0][0] * (mat[1][1] * mat[2][3] - mat[1][3] * mat[2][1]) -
					  mat[0][1] * (mat[1][0] * mat[2][3] - mat[1][3] * mat[2][0]) +
					  mat[0][3] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]);
		float det_all = mat[0][0] * (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1]) -
					    mat[0][1] * (mat[1][0] * mat[2][2] - mat[1][2] * mat[2][0]) +
					    mat[0][2] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]);
		float t = det_3 / det_all;

		vec f = ray.origin() + t * (-s) - p;
		float det_1 = f.x() * v.z() - v.x() * f.z();
		float det_2 = u.x() * f.z() - f.x() * u.z();
		float det_all2 = u.x() * v.z() - v.x() * u.z();
		float t1 = det_1 / det_all2;
		float t2 = det_2 / det_all2;

		if (t_min < t && t < t_max) {
			rec.t = t;
			rec.p = ray.point_at_parameter(t);
			rec.normal = normal;
			rec.ambient_color = vec(1, 1, 1) * (((unsigned(t1 * 50) / 10) % 2 + (unsigned(t2 * 50) / 10) % 2) % 2);
			//rec.ambient_color = rec.mat_ptr->ambient_color;
            rec.mat_ptr = mat_ptr;
			return true;
		}

		return false;
	}
	
	slab bound() const {
		return slab(vec(-INF, -INF, -INF), vec(INF, INF, INF));
	}
};

class Triangle: public Hittable {
public:
	vec p; // Origin point
	vec u, v; // Direction vectors
	vec normal;

	float a, b, c, d;
    material *mat_ptr = nullptr;
    
    bool proj_x, proj_y, proj_z;



	vec force_color = vec(-1000, 0, 0);

	Triangle() {}
	~Triangle() {
		if (mat_ptr != nullptr) {
			delete mat_ptr;
		}
	}
	Triangle(
        const vec &A, const vec &B, const vec &C, material* const mat_ptr) :
			mat_ptr(mat_ptr) {

		// a, b, c - in clockwise order to camera
		
		vec point = B;
		vec dir1 = A - B;
		vec dir2 = C - B;
		p = point;
		u = dir1;
		v = dir2;

		float tx = v.z() * u.y() - v.y() * u.z();
		a = -u.y() * tx;
		float ty = u.x() * v.z() - u.z() * v.x();
		b = u.y() * ty;
		float tz = v.x() * u.y() - v.y() * u.x();
		c = u.y() * tz;
		d = (p.x() * u.y() - p.y() * u.x()) * tx -
		    (p.z() * u.y() - p.y() * u.z()) * tz;

		normal = (vec(tx, -ty, -tz)).unit();

        vec test_u = vec(0, u.y(), u.z());
		vec test_v = vec(0, v.y(), v.z());
		
		proj_x = !(test_u.len2() < EPS || test_v.len2() < EPS ||
				(fabs(dot(test_u.unit(), test_v.unit())) > 1 - EPS));
        
        test_u = vec(u.x(), 0, u.z());
		test_v = vec(v.x(), 0, v.z());
		
		proj_y = !(test_u.len2() < EPS || test_v.len2() < EPS ||
				(fabs(dot(test_u.unit(), test_v.unit())) > 1 - EPS));
		
		test_u = vec(u.x(), u.y(), 0);
		test_v = vec(v.x(), v.y(), 0);
		
		proj_z = !(test_u.len2() < EPS || test_v.len2() < EPS ||
				(fabs(dot(test_u.unit(), test_v.unit())) > 1 - EPS));
	}

	bool hit(const Ray &ray, float t_min, float t_max, hit_record &rec) const {
		vec ray_pos = ray.origin() - p;
		if (dot(ray_pos, normal) > 0) {			
			if (dot(ray.direction(), normal) >= 0) {
				return false;
			}
		} else {
			if (dot(ray.direction(), normal) <= 0) {
				return false;
			}
		}
		
		// M = p + t_1*u + t_2*v = q + t_3*s =>
		// t_1*u + t_2*v - t_3*s = q - p
		float t;
		vec f;
		if (rec.pass_arg) {
			t = rec.t;
			f = rec.p - p;
		} else {
			vec s = -ray.direction();
			vec origin = ray.origin() - p;
			float mat[3][4] = {
				{u.x(), v.x(), s.x(), origin.x()},
				{u.y(), v.y(), s.y(), origin.y()},
				{u.z(), v.z(), s.z(), origin.z()}
			};
			float det_3 = mat[0][0] * (mat[1][1] * mat[2][3] - mat[1][3] * mat[2][1]) -
						  mat[0][1] * (mat[1][0] * mat[2][3] - mat[1][3] * mat[2][0]) +
						  mat[0][3] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]);
			float det_all = mat[0][0] * (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1]) -
							mat[0][1] * (mat[1][0] * mat[2][2] - mat[1][2] * mat[2][0]) +
							mat[0][2] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]);
			t = det_3 / det_all;
			f = ray.point_at_parameter(t) - p;
		}
		
		float det_1, det_2, det_all2;
		
		if (proj_y) {
			det_1 = f.x() * v.z() - v.x() * f.z();
			det_2 = u.x() * f.z() - f.x() * u.z();
			det_all2 = u.x() * v.z() - v.x() * u.z();
		} else if (proj_z) {
			det_1 = f.x() * v.y() - v.x() * f.y();
			det_2 = u.x() * f.y() - f.x() * u.y();
			det_all2 = u.x() * v.y() - v.x() * u.y();
		} else {
			det_1 = f.y() * v.z() - v.y() * f.z();
			det_2 = u.y() * f.z() - f.y() * u.z();
			det_all2 = u.y() * v.z() - v.y() * u.z();
		}
		float t1 = det_1 / det_all2;
		float t2 = det_2 / det_all2;

		if ((eq(t1, 0) || t1 > 0) && 
			(eq(t2, 0) || t2 > 0) &&
			(eq(t1, 1 - t2) || t1 < 1 - t2) &&
				t_min < t && t < t_max) {
			rec.t = t;
			rec.p = ray.point_at_parameter(t);
			rec.normal = normal;
            rec.mat_ptr = mat_ptr;
            rec.ambient_color = mat_ptr->ambient_color;
            if (force_color[0] >= 0) {
				rec.force_color = force_color;
			}
			return true;
		}

		return false;
	}
	
	slab bound() const {
		return slab(
				vec_min(p, vec_min(p + u, p + v)) - vec(EPS, EPS, EPS) * 3,
				vec_max(p, vec_max(p + u, p + v)) + vec(EPS, EPS, EPS) * 3
		);
	}
};

class TrianglePoly: public Hittable {
	shared_ptr<BVH_node> poly_root;
	
	vec p; // Origin point
	vec u, v; // Direction vectors
	vec normal;

	float a, b, c, d;
    material *mat_ptr = nullptr;
    
    bool proj_x, proj_y, proj_z;

public:
	vec force_color = vec(-1000, 0, 0);

	TrianglePoly() {}
	~TrianglePoly() {
		if (mat_ptr != nullptr) {
			delete mat_ptr;
		}
	}
	TrianglePoly(
        shared_ptr<Triangle> example_tri,
        shared_ptr<BVH_node> root,
        material* const mat_ptr) :
			mat_ptr(mat_ptr),
			poly_root(root) {

		// a, b, c - in clockwise order to camera
		
		vec point = example_tri->p;
		vec dir1 = example_tri->u;
		vec dir2 = example_tri->v;
		p = point;
		u = dir1;
		v = dir2;

		float tx = v.z() * u.y() - v.y() * u.z();
		a = -u.y() * tx;
		float ty = u.x() * v.z() - u.z() * v.x();
		b = u.y() * ty;
		float tz = v.x() * u.y() - v.y() * u.x();
		c = u.y() * tz;
		d = (p.x() * u.y() - p.y() * u.x()) * tx -
		    (p.z() * u.y() - p.y() * u.z()) * tz;

		normal = (vec(tx, -ty, -tz)).unit();

        vec test_u = vec(0, u.y(), u.z());
		vec test_v = vec(0, v.y(), v.z());
		
		proj_x = !(test_u.len2() < EPS || test_v.len2() < EPS ||
				(fabs(dot(test_u.unit(), test_v.unit())) > 1 - EPS));
        
        test_u = vec(u.x(), 0, u.z());
		test_v = vec(v.x(), 0, v.z());
		
		proj_y = !(test_u.len2() < EPS || test_v.len2() < EPS ||
				(fabs(dot(test_u.unit(), test_v.unit())) > 1 - EPS));
		
		test_u = vec(u.x(), u.y(), 0);
		test_v = vec(v.x(), v.y(), 0);
		
		proj_z = !(test_u.len2() < EPS || test_v.len2() < EPS ||
				(fabs(dot(test_u.unit(), test_v.unit())) > 1 - EPS));
	}

	bool hit(const Ray &ray, float t_min, float t_max, hit_record &rec) const {
		// M = p + t_1*u + t_2*v = q + t_3*s =>
		// t_1*u + t_2*v - t_3*s = q - p
		vec s = -ray.direction();
		vec origin = ray.origin() - p;
		float mat[3][4] = {
			{u.x(), v.x(), s.x(), origin.x()},
			{u.y(), v.y(), s.y(), origin.y()},
			{u.z(), v.z(), s.z(), origin.z()}
		};
		float det_3 = mat[0][0] * (mat[1][1] * mat[2][3] - mat[1][3] * mat[2][1]) -
					  mat[0][1] * (mat[1][0] * mat[2][3] - mat[1][3] * mat[2][0]) +
					  mat[0][3] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]);
		float det_all = mat[0][0] * (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1]) -
					    mat[0][1] * (mat[1][0] * mat[2][2] - mat[1][2] * mat[2][0]) +
					    mat[0][2] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]);
		float t = det_3 / det_all;

		rec.t = t;
		rec.p = ray.point_at_parameter(t);
		//rec.force_color = force_color;
		rec.pass_arg = true;
		bool success = poly_root->hit(ray, t_min, t_max, rec);
		rec.pass_arg = false;
		return success;
	}
	
	slab bound() const {
		return poly_root->box;
	}
};

class Circle: public Hittable {
	vec p; // Origin point, also center of circle
	vec u, v; // Direction vectors
	vec normal;

	float r;
	float a, b, c, d;
    material *mat_ptr = nullptr;
    
    bool proj_x, proj_y, proj_z;

public:
	vec force_color = vec(-1000, 0, 0);

	Circle() {}
	~Circle() {
		if (mat_ptr != nullptr) {
			delete mat_ptr;
		}
	}
	Circle(
        const vec &C, const float r, const vec &u, const vec &v, material* const mat_ptr) :
			p(C), r(r), u(u), v(v), mat_ptr(mat_ptr) {

		// a, b, c - in clockwise order to camera

		float tx = v.z() * u.y() - v.y() * u.z();
		a = -u.y() * tx;
		float ty = u.x() * v.z() - u.z() * v.x();
		b = u.y() * ty;
		float tz = v.x() * u.y() - v.y() * u.x();
		c = u.y() * tz;
		d = (p.x() * u.y() - p.y() * u.x()) * tx -
		    (p.z() * u.y() - p.y() * u.z()) * tz;

		normal = (vec(tx, -ty, -tz)).unit();

        vec test_u = vec(0, u.y(), u.z());
		vec test_v = vec(0, v.y(), v.z());
		
		proj_x = !(test_u.len2() < EPS || test_v.len2() < EPS ||
				(fabs(dot(test_u.unit(), test_v.unit())) > 1 - EPS));
        
        test_u = vec(u.x(), 0, u.z());
		test_v = vec(v.x(), 0, v.z());
		
		proj_y = !(test_u.len2() < EPS || test_v.len2() < EPS ||
				(fabs(dot(test_u.unit(), test_v.unit())) > 1 - EPS));
		
		test_u = vec(u.x(), u.y(), 0);
		test_v = vec(v.x(), v.y(), 0);
		
		proj_z = !(test_u.len2() < EPS || test_v.len2() < EPS ||
				(fabs(dot(test_u.unit(), test_v.unit())) > 1 - EPS));
	}

	bool hit(const Ray &ray, float t_min, float t_max, hit_record &rec) const {
		if (dot(ray.origin() - p, normal) > 0) {			
			if (dot(ray.direction(), normal) >= 0) {
				return false;
			}
		} else {
			if (dot(ray.direction(), normal) <= 0) {
				return false;
			}
		}
		
		// M = p + t_1*u + t_2*v = q + t_3*s =>
		// t_1*u + t_2*v - t_3*s = q - p
		vec s = -ray.direction();
		vec origin = ray.origin() - p;
		float mat[3][4] = {
			{u.x(), v.x(), s.x(), origin.x()},
			{u.y(), v.y(), s.y(), origin.y()},
			{u.z(), v.z(), s.z(), origin.z()}
		};
		float det_3 = mat[0][0] * (mat[1][1] * mat[2][3] - mat[1][3] * mat[2][1]) -
					  mat[0][1] * (mat[1][0] * mat[2][3] - mat[1][3] * mat[2][0]) +
					  mat[0][3] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]);
		float det_all = mat[0][0] * (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1]) -
					    mat[0][1] * (mat[1][0] * mat[2][2] - mat[1][2] * mat[2][0]) +
					    mat[0][2] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]);
		float t = det_3 / det_all;

		vec f = ray.point_at_parameter(t) - p;
		
		if (f.len() < r &&
				t_min < t && t < t_max) {
			rec.t = t;
			rec.p = ray.point_at_parameter(t);
			rec.normal = normal;
            rec.mat_ptr = mat_ptr;
            rec.ambient_color = mat_ptr->ambient_color;
            if (force_color[0] >= 0) {
				rec.force_color = force_color;
			}
			return true;
		}

		return false;
	}
	
	slab bound() const {
		return slab(p - vec(r, r, r), p + vec(r, r, r));
	}
};

class Light {
public:
	vec pos, intensity;
	
	Light(vec pos, vec intensity) : pos(pos), intensity(intensity) {}
	vec get_pos() const { return pos; }
	vec get_color() const { return intensity; }
};

Image bkg;

vec bkg_color(const Ray &r) {
	vec v = r.direction().unit();
	vec u = vec(v.x(), 0, v.z()).unit();
	float psi = acos( dot(vec(0, 1, 0), v) );
	float phi = acos( dot(vec(1, 0, 0), u) );
	if ( dot(vec(0, 0, 1), u) > 0 ) {
		phi = M_PI * 2 - phi;
	}
	
	int h = psi * bkg.n_rows / M_PI;
	int w = phi * bkg.n_cols / (M_PI * 2);
	if (h < 0) {
		h = 0;
	}
	if (w < 0) {
		w = 0;
	}
	if (h < 0 or w < 0) {
		cout << h << " " << w << "\n";
		cout << "VIOLATION\n";
		exit(0);
	}
	return bkg.get(h, w) / 255;
	
    vec unit_direction = r.direction().unit();
    float t = 0.5*(unit_direction.y() + 1.0);
    return (1-t)*vec(0, 0, 0) + t*vec(0.5, 0.7, 1.0);
}

vec color(const Ray &ray, const HittableList &objects, const vector<Light> &lights, int recursion) {
    if (recursion > 4) {
		//printf("Exceeded\n");
		return bkg_color(ray);
	}

	vec col = bkg_color(ray);
	hit_record hit_rec;

	if (objects.hit(ray, EPS, INF, hit_rec)) {		
		vec normal = hit_rec.normal;
        vec vec_in = ray.direction().unit();
        float dot_i_n = dot(vec_in, normal);
        vec abs_normal = dot_i_n < 0 ? normal : -normal;
        float abs_dot_i_n = fabs(dot_i_n);
		
		if (hit_rec.force_color[0] >= 0) { // Force a specific color
			col = hit_rec.force_color;
		} else { // Color-code the normal to surface
			
			// DIFFUSE AND SPECULAR LIGHT
			vec col_diffuse(0, 0, 0);
			vec col_specular(0, 0, 0);
			for (int i = 0; i < lights.size(); ++i) {
				vec light_dir = lights[i].get_pos() - hit_rec.p;
				float light_dist = light_dir.len2();
				light_dir = light_dir.unit();
				
				hit_record hit_shadow;
				Ray light_ray = Ray(hit_rec.p, light_dir, ray.theta());
				bool is_shadow = objects.hit(light_ray, EPS, INF, hit_shadow) && 
						hit_shadow.t * hit_shadow.t < light_dist;
				
				if (!is_shadow) {
					float dot_l_n = fmax(0, dot(light_dir, abs_normal)); // Light cannot go through the surface
					col_diffuse += lights[i].get_color() * dot_l_n;
					
					vec light_reflect = reflect(-light_dir, abs_normal);
					col_specular += lights[i].get_color() * 
							powf(fmax(0.f, -dot(light_reflect, vec_in)), 25);
				}
			}
			const float &k_diffuse = hit_rec.mat_ptr->k_diffuse;
			vec col_ambient = (col_diffuse * hit_rec.ambient_color * k_diffuse +
					col_specular * (1 - k_diffuse)) / lights.size();
			
			if (hit_rec.mat_ptr->mat_type == MAT_DIFFUSE) {
				col = col_ambient;
			} else {
				
				// REFLECTION
				vec reflect_dir = vec_in + 2 * abs_normal * abs_dot_i_n;
                Ray reflect_ray = Ray(hit_rec.p, reflect_dir, ray.theta());
                vec col_reflect = color(reflect_ray, objects, lights, recursion + 1);
                
				if (hit_rec.mat_ptr->mat_type == MAT_METAL) {
					col = col_ambient * (1 - hit_rec.mat_ptr->k_mirror) + col_reflect * hit_rec.mat_ptr->k_mirror;
				} else {
					
					// REFRACTION
					
					float theta;
					float theta_sq;
					if (dot_i_n < 0) { // Ray enters body
						theta = hit_rec.mat_ptr->theta;
						theta_sq = hit_rec.mat_ptr->theta_sq;
					} else { // Ray exits body
						theta = hit_rec.mat_ptr->theta_inv;
						theta_sq = hit_rec.mat_ptr->theta_sq_inv;
					}

					vec u = vec_in + abs_normal * abs_dot_i_n;

					float cos_a = abs_dot_i_n;
					float sin_a_sq = 1 - cos_a * cos_a;

					float r0 = (1 - theta) / (1 + theta); // Shlick formula
					r0 = r0 * r0;
					float reflect_probability = r0 + (1 - r0) * pow(1 - cos_a, 5);
					reflect_probability += hit_rec.mat_ptr->k_reflect;
					if (reflect_probability < 0) {
						reflect_probability = 0;
					}
					if (reflect_probability > 1) {
						reflect_probability = 1;
					}

					if (sin_a_sq >= theta_sq) {
						// Total internal reflection
						col = col_ambient * (1 - hit_rec.mat_ptr->k_mirror) +
								col_reflect * hit_rec.mat_ptr->k_mirror;
					} else {
						// Refract through surface
						float sin_a = sqrt(sin_a_sq);
						vec refract_dir = -abs_normal * sqrt(theta_sq - sin_a_sq) +
								u.unit() * sin_a;

						Ray refract_ray(hit_rec.p, refract_dir, 1.0);
						vec col_refract = color(refract_ray, objects, lights, recursion + 1);
						col = col_ambient * (1 - hit_rec.mat_ptr->k_mirror) +
								(col_reflect * reflect_probability + 
								col_refract * (1 - reflect_probability)) * hit_rec.mat_ptr->k_mirror;
					}
				}
			}
		}
	} else {
		col = bkg_color(ray);
	}
	return col;
}


void ray_caster(
    Image &img, int row_begin, int row_end,
    HittableList &objects, const vector<Light> &lights,
    const Camera &main_camera, int n, int m) {

    for (int i = row_begin; i < row_end; i++) {
		for (int j = 0; j < m; j++) {
			vec col(0, 0, 0);
			int antialias = 1;
			for (int its = 0; its < antialias; its++) {
				float v = (float(n - i) + random_double()*0) / float(n);
				float u = (float(j) + random_double()*0) / float(m);
				Ray r = main_camera.get_ray(u, v);
				col += color(r, objects, lights, 0);
			}
			col = col / antialias;

			for (int c = 0; c < 3; c++) {
				if (col[c] < 0) {
					col[c] = 0;
				}
				if (col[c] > 1) {
					col[c] = 1;
				}
			}

			if (col[0] < 0.0 || col[0] > 1.0 ||
					col[1] < 0.0 || col[1] > 1.0 ||
					col[2] < 0.0 || col[2] > 1.0) {
				cout << "Range violation: " << col << "\n";
				printf("Color out of [0.0, 1.0] range! Exiting...\n");
				exit(0);
			}

			img.set(i, j, vec(
					powf(col[0], 0.5) * 255,
					powf(col[1], 0.5) * 255,
					powf(col[2], 0.5) * 255)
			);
		}
	}
}



void read_poly(HittableList &objects, const char* filename, vec offset, float *rotate_mat, float scale) {
	vector< shared_ptr<Hittable> > tri_list;
	shared_ptr<Triangle> example;
	vector< shared_ptr<Hittable> > tri_list2;
	shared_ptr<Triangle> example2;
	vector< shared_ptr<Hittable> > tri_list3;
	
	material *mat = new material(MAT_DIFFUSE);
	mat->k_diffuse = 0.9;
	
	vec A, B, C;
	
	std::ifstream fin;
	fin.open(filename, std::ifstream::in);
	
	// Create two planes
	int c;
	fin >> c;
	vec backz = vec(0, 0, -0.1) * scale;
	for (int i = 0; i < c; i++) {
		fin >> A[0] >> A[1] >> A[2];
		fin >> B[0] >> B[1] >> B[2];
		fin >> C[0] >> C[1] >> C[2];
		A = A * scale;
		B = B * scale;
		C = C * scale;
		
		vec D = A + backz;
		vec E = B + backz;
		vec F = C + backz;
		
		A = mult_v_mat(A, rotate_mat) + offset;
		B = mult_v_mat(B, rotate_mat) + offset;
		C = mult_v_mat(C, rotate_mat) + offset;
		
		D = mult_v_mat(D, rotate_mat) + offset;
		E = mult_v_mat(E, rotate_mat) + offset;
		F = mult_v_mat(F, rotate_mat) + offset;
		
		
		tri_list.push_back( make_shared<Triangle>(A, B, C, mat->clone()) );
		if (i == 0) {
			example = make_shared<Triangle>(A, B, C, mat->clone());
		}
		
		tri_list2.push_back( make_shared<Triangle>(D, E, F, mat->clone()) );
		if (i == 0) {
			example2 = make_shared<Triangle>(D, E, F, mat->clone());
		}
	}
	
	// Sides
	fin >> c;
	for (int i = 0; i < c; i++) {
		fin >> A[0] >> A[1] >> A[2];
		fin >> B[0] >> B[1] >> B[2];
		A = A * scale;
		B = B * scale;
		
		vec A2 = A + backz;
		vec B2 = B + backz;
		
		A = mult_v_mat(A, rotate_mat) + offset;
		B = mult_v_mat(B, rotate_mat) + offset;
		
		A2 = mult_v_mat(A2, rotate_mat) + offset;
		B2 = mult_v_mat(B2, rotate_mat) + offset;
		
		tri_list3.push_back( make_shared<Triangle>(A, A2, B, mat->clone()) );
		tri_list3.push_back( make_shared<Triangle>(B, A2, B2, mat->clone()) );
	}
	
	shared_ptr<BVH_node> tree_root = make_shared<BVH_node>(tri_list);
	shared_ptr<TrianglePoly> tri_face = make_shared<TrianglePoly>(example, tree_root, mat->clone());
	objects.add(tree_root);
	
	shared_ptr<BVH_node> tree_root2 = make_shared<BVH_node>(tri_list2);
	shared_ptr<TrianglePoly> tri_face2 = make_shared<TrianglePoly>(example2, tree_root2, mat->clone());
	//objects.add(tree_root2);
	
	shared_ptr<BVH_node> tree_root3 = make_shared<BVH_node>(tri_list3);
	objects.add(tree_root3);
}

void scene_3(HittableList &objects, vector<Light> &lights) {
	lights.push_back(Light(vec(3, 1, -4), vec(0.8, 0.4, 0.2)));
	lights.push_back(Light(vec(-2, 2, -2), vec(1, 0.3, 0.05)));
	
	material *red_dif = new material(MAT_DIFFUSE);
	red_dif->ambient_color = vec(1, 0.5, 1);
	red_dif->k_diffuse = 0.8;
	
	material *red_metal = new material(MAT_METAL, 1.5);
	red_metal->k_mirror = 0.7;
	red_metal->ambient_color = vec(1, 0, 0.3);
	
	vector< shared_ptr<Hittable> > regular;
	vector< shared_ptr<Hittable> > special;
	
	special.push_back( make_shared<Circle>(vec(0, 0, 0), 10, vec(1, 0, 0), vec(0, 0, -1), red_metal->clone()) );
	
	float mat_x[9];
	float mat_y[9];
	float mat_z[9];
	float phi;
	float scale;
	
	// Letter 0
	rot_x(-0 * M_PI / 180, mat_x);
	rot_y(0 * M_PI / 180, mat_y);
	rot_z(0, mat_z);
	scale = 4.4;
	read_poly(objects, "letters/letter-0-list.in", vec(-1 * scale, (0.5 - 140.0 / 800.0) * scale, -5), mat_x, scale);
	
	// Letter 1
	rot_x(-0 * M_PI / 180, mat_x);
	rot_y(0 * M_PI / 180, mat_y);
	rot_z(0, mat_z);
	read_poly(objects, "letters/letter-4-list.in", vec(0 * scale, (0.5 - 140.0 / 800.0) * scale, -5), mat_x, scale);
	
	// Letter 2
	rot_x(-0 * M_PI / 180, mat_x);
	rot_y(0 * M_PI / 180, mat_y);
	rot_z(0, mat_z);
	read_poly(objects, "letters/letter-2-list.in", vec(1 * scale, (0.5 - 140.0 / 800.0) * scale, -5), mat_x, scale);
	
	// Letter 3
	rot_x(-0 * M_PI / 180, mat_x);
	rot_y(0 * M_PI / 180, mat_y);
	rot_z(0, mat_z);
	scale = 1.4;
	read_poly(objects, "letters/letter-3-list.in", vec(-2.5 * scale, (0.5 - 116.0 / 800.0) * scale, -3), mat_x, scale);
	
	// Letter 4
	rot_x(-0 * M_PI / 180, mat_x);
	rot_y(0 * M_PI / 180, mat_y);
	rot_z(0, mat_z);
	scale = 1.4;
	read_poly(objects, "letters/letter-4-list.in", vec(-1.5 * scale, (0.5 - 124.0 / 800.0) * scale, -3), mat_x, scale);
	
	// Letter 5
	rot_x(-0 * M_PI / 180, mat_x);
	rot_y(0 * M_PI / 180, mat_y);
	rot_z(0, mat_z);
	scale = 1.4;
	read_poly(objects, "letters/letter-5-list.in", vec(-0.5 * scale, (0.5 - 103.0 / 800.0) * scale, -3), mat_x, scale);
	
	// Letter 6
	rot_x(-0 * M_PI / 180, mat_x);
	rot_y(0 * M_PI / 180, mat_y);
	rot_z(0, mat_z);
	scale = 1.4;
	read_poly(objects, "letters/letter-6-list.in", vec(0.5 * scale, (0.5 - 103.0 / 800.0) * scale, -3), mat_x, scale);
	
	// Letter 7
	rot_x(-0 * M_PI / 180, mat_x);
	rot_y(0 * M_PI / 180, mat_y);
	rot_z(0, mat_z);
	scale = 1.4;
	read_poly(objects, "letters/letter-3-list.in", vec(1.5 * scale, (0.5 - 116.0 / 800.0) * scale, -3), mat_x, scale);
	
	// Letter 8
	rot_x(-0 * M_PI / 180, mat_x);
	rot_y(0 * M_PI / 180, mat_y);
	rot_z(0, mat_z);
	scale = 1.4;
	read_poly(objects, "letters/letter-8-list.in", vec(2.5 * scale, (0.5 - 116.0 / 800.0) * scale, -3), mat_x, scale);
	
	// Letter 9
	rot_x(-0 * M_PI / 180, mat_x);
	rot_y(0 * M_PI / 180, mat_y);
	rot_z(0, mat_z);
	scale = 0.5;
	read_poly(objects, "letters/letter-9-list.in", vec(-1 * scale, (0.5 - 116.0 / 800.0) * scale, -2), mat_x, scale);
	
	// Letter 10
	rot_x(-0 * M_PI / 180, mat_x);
	rot_y(0 * M_PI / 180, mat_y);
	rot_z(0, mat_z);
	scale = 0.5;
	read_poly(objects, "letters/letter-4-list.in", vec(0 * scale, (0.5 - 116.0 / 800.0) * scale, -2), mat_x, scale);
	
	// Letter 11
	rot_x(-0 * M_PI / 180, mat_x);
	rot_y(0 * M_PI / 180, mat_y);
	rot_z(0, mat_z);
	scale = 0.5;
	read_poly(objects, "letters/letter-11-list.in", vec(1 * scale, (0.5 - 116.0 / 800.0) * scale, -2), mat_x, scale);
	
	
	
	
	
	
	
	// Letter 12
	phi = (80 + 20) * M_PI / 180;
	rot_x(-90 * M_PI / 180, mat_x);
	rot_y(10 * M_PI / 180, mat_y);
	rot_z(-M_PI/2 + phi, mat_z);
	mult_mat(mat_x, mat_y);
	mult_mat(mat_x, mat_z);
	read_poly(objects, "letters/letter-12-list.in", vec(cos(phi) * 1.5, 0.05, -sin(phi) * 1), mat_x, 0.4);
	
	// Letter 13
	phi = (80) * M_PI / 180;
	rot_x(-90 * M_PI / 180, mat_x);
	rot_y(-10 * M_PI / 180, mat_y);
	rot_z(-M_PI/2 + phi, mat_z);
	mult_mat(mat_x, mat_y);
	mult_mat(mat_x, mat_z);
	read_poly(objects, "letters/letter-0-list.in", vec(cos(phi) * 1.5, 0.05, -sin(phi) * 1), mat_x, 0.4);
	
	
	
	
	
	
	
	float st = 20;
	float da = (180.0 - st * 2) / 8;
	float dh = da / 2;
	
	// Letter 18
	phi = (st + da * 0 + dh) * M_PI / 180;
	rot_x(-90 * M_PI / 180, mat_x);
	rot_y(-30 * M_PI / 180, mat_y);
	rot_z(0, mat_z);
	mult_mat(mat_x, mat_y);
	mult_mat(mat_x, mat_z);
	read_poly(objects, "letters/letter-9-list.in", vec(1, 0.3, -sin(phi) * 1), mat_x, 0.5);
	
	// Letter 17
	phi = (st + da * 0 + dh) * M_PI / 180;
	rot_x(-90 * M_PI / 180, mat_x);
	rot_y(-15 * M_PI / 180, mat_y);
	rot_z(0, mat_z);
	mult_mat(mat_x, mat_y);
	mult_mat(mat_x, mat_z);
	read_poly(objects, "letters/letter-2-list.in", vec(0.5, 0.1, -sin(phi) * 1), mat_x, 0.5);
	
	// Letter 16
	phi = (st + da * 0 + dh) * M_PI / 180;
	rot_x(-90 * M_PI / 180, mat_x);
	rot_y(0 * M_PI / 180, mat_y);
	rot_z(0, mat_z);
	mult_mat(mat_x, mat_y);
	mult_mat(mat_x, mat_z);
	read_poly(objects, "letters/letter-16-list.in", vec(0, 0.05, -sin(phi) * 1), mat_x, 0.5);
	
	// Letter 15
	phi = (st + da * 0 + dh) * M_PI / 180;
	rot_x(-90 * M_PI / 180, mat_x);
	rot_y(15 * M_PI / 180, mat_y);
	rot_z(0, mat_z);
	mult_mat(mat_x, mat_y);
	mult_mat(mat_x, mat_z);
	read_poly(objects, "letters/letter-0-list.in", vec(-0.5, 0.1, -sin(phi) * 1), mat_x, 0.5);
	
	// Letter 14
	phi = (st + da * 0 + dh) * M_PI / 180;
	rot_x(-90 * M_PI / 180, mat_x);
	rot_y(30 * M_PI / 180, mat_y);
	rot_z(0, mat_z);
	mult_mat(mat_x, mat_y);
	mult_mat(mat_x, mat_z);
	read_poly(objects, "letters/letter-4-list.in", vec(-1, 0.3, -sin(phi) * 1), mat_x, 0.5);
	
	
	
	
	// Letter 3
	phi = (30 + 20 * 2 + 10) * M_PI / 180;
	rot_x(-80 * M_PI / 180, mat_x);
	rot_y(0, mat_y);
	rot_z(-M_PI/2 + (phi + 30 * M_PI / 180), mat_z);
	mult_mat(mat_x, mat_y);
	mult_mat(mat_x, mat_z);
	
	read_poly(objects, "letters/letter-16-list.in", vec(cos(phi) * 1.5, -0.9, -sin(phi) * 1.5), mat_x, 1);
	
	// Letter 2
	phi = (30 + 20 * 3 + 10) * M_PI / 180;
	rot_x(-90 * M_PI / 180, mat_x);
	rot_y(0, mat_y);
	rot_z(-M_PI/2 + (phi - 20 * M_PI / 180), mat_z);
	mult_mat(mat_x, mat_y);
	mult_mat(mat_x, mat_z);
	
	read_poly(objects, "letters/letter-0-list.in", vec(cos(phi) * 3, -0.9, -sin(phi) * 3), mat_x, 1);
	
	// Letter 1
	phi = (30 + 20 * 4 + 10) * M_PI / 180;
	rot_x(0 * M_PI / 180, mat_x);
	rot_y(-M_PI/2 + phi, mat_y);
	rot_z(0, mat_z);
	mult_mat(mat_x, mat_y);
	mult_mat(mat_x, mat_z);
	
	read_poly(objects, "letters/letter-4-list.in", vec(cos(phi) * 3, -0.66, -sin(phi) * 3), mat_x, 1);
	
	// Letter 0
	phi = (30 + 20 * 5 + 10) * M_PI / 180;
	rot_x(-90 * M_PI / 180, mat_x);
	rot_y(0, mat_y);
	rot_z(-M_PI/2 + phi, mat_z);
	mult_mat(mat_x, mat_y);
	mult_mat(mat_x, mat_z);
	
	read_poly(objects, "letters/letter-12-list.in", vec(cos(phi) * 3, -0.9, -sin(phi) * 3), mat_x, 1);
	
	
	phi = (30 + 20 * 0 + 10) * M_PI / 180;
	rot_x(-90 * M_PI / 180, mat_x);
	rot_y(0 * M_PI / 180, mat_y);
	rot_z(-M_PI/2 + phi, mat_z);
	mult_mat(mat_x, mat_y);
	mult_mat(mat_x, mat_z);
	
	//read_poly(objects, "letters/letter-5-list.in", vec(cos(phi) * 3, -0.9, -sin(phi) * 3), mat_x);
	
	
	
	
	shared_ptr<BVH_node> spec_root = make_shared<BVH_node>(special);
	objects.add(spec_root);
}

void scene_2(HittableList &objects, vector<Light> &lights) {
	lights.push_back(Light(vec(0, 0, 0.3), vec(1, 1, 1)));
	
	material *red_dif = new material(MAT_DIFFUSE);
	red_dif->ambient_color = vec(0.9, 0.1, 0.2);
	red_dif->k_diffuse = 0.8;
	
	material *blue_dif = new material(MAT_DIFFUSE);
	blue_dif->ambient_color = vec(0.1, 0.2, 1);
	blue_dif->k_diffuse = 1.0;
	
	material *violet_dif = new material(MAT_DIFFUSE);
	violet_dif->ambient_color = vec(0.6, 0.1, 1);
	violet_dif->k_diffuse = 1.0;
	
	material *cyan_dif = new material(MAT_DIFFUSE);
	cyan_dif->ambient_color = vec(0.1, 0.8, 0.8);
	cyan_dif->k_diffuse = 1.0;
	
	material *white_dif = new material(MAT_DIFFUSE);
	white_dif->ambient_color = vec(1, 1, 1);
	white_dif->k_diffuse = 1.0;
	
	material *red_metal = new material(MAT_METAL, 1.5);
	red_metal->k_diffuse = 1.0;
	red_metal->k_mirror = 0.7;
	red_metal->ambient_color = vec(1, 0, 0.3);
	
	material *blue_metal = new material(MAT_METAL, 1.5);
	blue_metal->k_diffuse = 0.1;
	blue_metal->k_mirror = 0.7;
	blue_metal->ambient_color = vec(0, 0.2, 1);
	
	material *clear_glass = new material(MAT_DIELECTRIC, 2.4);
	clear_glass->k_diffuse = 0.7;
	clear_glass->k_mirror = 1;
	clear_glass->k_reflect = 0.2;
	
	material *red_glass = new material(MAT_DIELECTRIC, 1.6);
	red_glass->k_diffuse = 0.7;
	red_glass->k_mirror = 0.5;
	red_glass->k_reflect = 0.1;
	red_glass->ambient_color = vec(1, 0, 0);
	
	material* mat_arr[] = {red_dif, blue_dif, violet_dif, cyan_dif, white_dif};
	
	vector< shared_ptr<Hittable> > special;
	vector< shared_ptr<Hittable> > regular;
	
	float phi = 0;
	for (float z = -4; z > -100; z -= 0.5) {
		float x = cos(phi) * 6;
		float y = sin(phi) * 6;
		phi += rad(12);
		
		int mat_idx = random_double() * (sizeof(mat_arr) / sizeof(*mat_arr) - 1);
		regular.push_back( make_shared<Sphere>(vec(x, y, z), 0.6, mat_arr[mat_idx]->clone()) );
	}
	
	
	regular.push_back( make_shared<Sphere>(vec(0, -0.3, -2.5), 0.5, clear_glass->clone()) );
	regular.push_back( make_shared<Sphere>(vec(1, -0.1, -3.5), 0.5, red_metal->clone()) );
	regular.push_back( make_shared<Sphere>(vec(-1, -0.1, -1.5), 0.5, blue_metal->clone()) );
	
	special.push_back( make_shared<InfinitePlane>(vec(1000, -1, 0), vec(1, 0, 0), vec(0, 0, 1), cyan_dif->clone()) );
	
	shared_ptr<BVH_node> reg_root = make_shared<BVH_node>(regular);
	objects.add(reg_root);
		
	shared_ptr<BVH_node> spec_root = make_shared<BVH_node>(special);
	objects.add(spec_root);
}

void main_wrapped(Image *img_ptr, int threads, int thr, float fov_h, float fov_v, int n_img, int m_img) {
	int n_start = n_img / threads * thr;
	int n_end;
	if (thr == threads - 1) {
		n_end = n_img;
	} else {
		n_end = n_start + n_img / threads;
	}
	
	Image img_all(n_img, m_img);
	HittableList objects;
	vector<Light> lights;
	Camera main_camera(fov_h, fov_v);
	
	int scene = 1;
	if (scene == 1) {
		EPS = 1e-5;
		bkg.load("panorama.bmp");
		main_camera = Camera(fov_h, fov_v, -15, vec(0.0, 1.0, 0.5));
		scene_3(objects, lights);
	} else {
		EPS = 1e-3;
		bkg.load("panorama2.bmp");
		main_camera = Camera(fov_h, fov_v);
		scene_2(objects, lights);
	}
	
	// Run scene
	objects.construct_BVH();
	ray_caster(img_all, n_start, n_end, objects, lights, main_camera, n_img, m_img);
	
	// Save results
	if (threads == 1) {
		std::string path("hi.bmp");
		img_all.save(path);
	} else {
		int mem_begin = n_start * (m_img * 3);
		int mem_end = n_end * (m_img * 3);
		memcpy(img_ptr->image + mem_begin, img_all.image + mem_begin, mem_end - mem_begin);
	}
}

int main()
{
	vector<std::thread> pool;
	vector<Image*> pool_buf;
	int threads = 1;
	
	float fov_h = 105 * M_PI / 180;
	float fov_v = 80 * M_PI / 180;
	float dpi = 900;
	int n_img = tan(fov_v / 2) * 2 * dpi;
	int m_img = tan(fov_h / 2) * 2 * dpi;
	
	Image img_all(n_img, m_img);
	for (int thr = 0; thr < threads; thr++) {
		pool_buf.push_back(new Image(n_img, m_img));
		pool.push_back(
		std::thread(main_wrapped, pool_buf[thr], threads, thr, fov_h, fov_v, n_img, m_img) );
	}
	
	for (int thr = 0; thr < threads; thr++) {
		int n_start = n_img / threads * thr;
		int n_end;
		if (thr == threads - 1) {
			n_end = n_img;
		} else {
			n_end = n_start + n_img / threads;
		}
		
		pool[thr].join();
		img_all.merge(*(pool_buf[thr]), n_start, n_end);
	}
    //cout << "Average BVH call: " << (float) depth / all_invokes << "\n";
	
	if (threads != 1) {
		std::string path("hi.bpm");
		img_all.save(path);
	}

    return 0;
}
