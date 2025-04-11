#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include <thread>

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "Bitmap.h"

using std::cout;
using std::vector;
using std::shared_ptr;
using std::make_shared;
using std::thread;

const float EPS = 1e-5;
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

// ******************************************************************* //
// ***************************** Image saving ************************ //
// ******************************************************************* //


class Image {
	const int COLOR_DEPTH = 3;
	unsigned char *image;
	int n_rows, n_cols, n_colors;

public:
	Image(int rows, int cols) {
		printf("Create\n");
		image = new unsigned char[rows * cols * COLOR_DEPTH];
		n_rows = rows;
		n_cols = cols;
		n_colors = COLOR_DEPTH;
	}

	unsigned char& operator()(int row, int col, int color) {
		return image[row * (n_cols * n_colors) + col * n_colors + color];
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

	void save(const char *path) {
		unsigned int *buf = new unsigned int[n_rows * n_cols];
		int idx = 0;
		for (int y = n_rows - 1; y >= 0; y--) {
			for (int x = 0; x < n_cols; x++) {
				buf[y * n_cols + x] = (image[idx + 2] << 16) + (image[idx + 1] << 8) + image[idx];
				//printf("%u %u %u %u\n", buf[y * n_cols + x], image[idx], image[idx+1], image[idx+2]);

				idx += n_colors;
			}
		}

		SaveBMP(path, buf, n_cols, n_rows);
		delete buf;

		/*FILE *fout = fopen(path, "w");
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
		fclose(fout);*/
	}

	~Image() {
		printf("Destroy\n");
		delete image;
	}
};



// ******************************************************************* //
// *************************** Basic geometry ************************ //
// ******************************************************************* //

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
	Camera(float horizontal_rad_fov=M_PI, float vertical_rad_fov=M_PI) {

		lower_left_corner = vec(-2.0, -1.0, -1.0);
		horizontal = vec(4.0, 0.0, 0.0);
		vertical = vec(0.0, 2.0, 0.0);
		origin = vec(0.0, 0.0, 0.0);
	}
	Ray get_ray(float u, float v) const {
		return Ray(origin, lower_left_corner + u*horizontal + v*vertical - origin, 1.0);
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
			BVH_node(list, 0, list.size(), 0) {}
	BVH_node(vector< shared_ptr<Hittable> > &objects, int beg_idx, int end_idx, int recs) {
		//for (int i = 0; i < recs*2; i++) {
		//	cout << " ";
		//}
		//cout << "Len: " << objects.size() << " from " << beg_idx << " to " << end_idx << "...\n";
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
		} else {
			std::sort(objects.begin() + beg_idx, objects.begin() + end_idx, comparator);

			int mid = (beg_idx + end_idx) / 2;
			left = make_shared<BVH_node>(objects, beg_idx, mid, recs + 1);
			right = make_shared<BVH_node>(objects, mid, end_idx, recs+1);
			box = surround_slab(left->bound(), right->bound());
		}
		
		//for (int i = 0; i < recs*2; i++) {
		//	cout << " ";
		//}
		//cout << "Box: " << box.min << " " << box.max << "\n\n";
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
	vector< shared_ptr<Hittable> > l; // Without pointer it won't compile
	BVH_node *tree_root;

public:
	HittableList() {}
	HittableList(const vector< shared_ptr<Hittable> > &list) { l = list; }
	~HittableList() {
		if (tree_root != nullptr) {
			delete tree_root;
		}
	}
	
	//vector< shared_ptr<Hittable> >& get_list() { return l; }

    void add(const shared_ptr<Hittable> &object) {
        l.push_back(object);
    }
    
    void construct_BVH() {
		cout << "Creating BVH\n";
		tree_root = new BVH_node(l);
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
    /*hit_record temp_rec;
    bool hit_anything = false;
    double closest_so_far = t_max;
    for (int i = 0; i < l.size(); i++) {
        if (l[i]->hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;*/
    
    return tree_root->hit(r, t_min, t_max, rec);
}


enum {
	MAT_DIFFUSE, MAT_METAL, MAT_DIELECTRIC
};

struct material {
	int mat_type = MAT_DIFFUSE;
	// color = (diffuse + specular) * (1 - k_mirror) +
	// + (reflect * k_reflect + refract * k_refract) * k_mirror
	float k_diffuse = 0.5;
	float k_specular = 1 - 0.5;
	
	float k_mirror = 1.0;
	
	float k_reflect = 0.1;
	float k_refract = 1 - 0.1;
	
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
			//rec.ambient_color = vec(1, 1, 1) * (((unsigned(t1 * 50) / 10) % 2 + (unsigned(t2 * 50) / 10) % 2) % 2);
			rec.ambient_color = vec(1, 1, 1);
            rec.mat_ptr = mat_ptr;
			return true;
		}

		return false;
	}
	
	slab bound() const {
		return slab(vec(-INF, -INF, -INF), vec(INF, INF, INF));
	}
};

int out1 = 0;

class Triangle: public Hittable {
	vec p; // Origin point
	vec u, v; // Direction vectors
	vec normal;

	float a, b, c, d;
    material *mat_ptr = nullptr;
    
    bool proj_x, proj_y, proj_z;

public:
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

		if (t1 > 0 && t2 > 0 && t1 < 1 - t2 &&
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
				vec_min(p, vec_min(p + u, p + v)),
				vec_max(p, vec_max(p + u, p + v))
		);
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

vec bkg_color(const Ray &r) {
	//return vec(0, 0, 0);
    vec unit_direction = r.direction().unit();
    float t = 0.5*(unit_direction.y() + 1.0);
    return (1-t)*vec(1.0, 1.0, 1.0) + t*vec(0.5, 0.7, 1.0);
}

int reflections = 0;
int outa=0;

vec color(const Ray &ray, const HittableList &objects, const vector<Light> &lights, int recursion) {
    if (recursion > 10) {
		//printf("Exceeded\n");
		return bkg_color(ray);
	}

	vec col = bkg_color(ray);
	hit_record hit_rec;

	if (objects.hit(ray, EPS, INF, hit_rec)) {
		reflections++;
		
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
			float k_diffuse = 0.5;
			float k_specular = 1 - k_diffuse;
			vec col_ambient = (col_diffuse * hit_rec.ambient_color * hit_rec.mat_ptr->k_diffuse +
					col_specular * hit_rec.mat_ptr->k_specular) / lights.size();
			
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
    const Camera &main_camera, int n, int m)
{
	//row_end = 1;
	//m = 1;
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


void addobj(HittableList &objects, vec a, vec b, vec c, material *mat) {
	objects.add( make_shared<Triangle>(a, b, c, mat) );
}

void addcube(HittableList &objects, vec shift_z, material *mat) {
	vec a, b, c, d, e, f, g, h;
    a = vec(0, 0, 0)*2 + shift_z;
    b = vec(0, 1, 0)*2 + shift_z;
    c = vec(1, 1, 0)*2 + shift_z;
    d = vec(1, 0, 0)*2 + shift_z;
    e = vec(0, 0, -1)*2 + shift_z;
    f = vec(0, 1, -1)*2 + shift_z;
    g = vec(1, 1, -1)*2 + shift_z;
    h = vec(1, 0, -1)*2 + shift_z;
    addobj(objects, a, b, c, mat);
    addobj(objects, a, c, d, mat);
    
    addobj(objects, g, f, e, mat);
    addobj(objects, h, g, e, mat);
    
    addobj(objects, d, c, g, mat);
    addobj(objects, d, g, h, mat);
    
    addobj(objects, f, b, a, mat);
    addobj(objects, f, a, e, mat);
    
    addobj(objects, d, h, a, mat);
    addobj(objects, h, e, a, mat);
    
    addobj(objects, c, b, f, mat);
    addobj(objects, c, f, g, mat);
}

int main()
{
	int n = 1000;
	int m = 2000;
	Image img(n, m);
	Camera main_camera;
	HittableList objects;
    vector<Light> lights;
    
    //lights.push_back(Light(vec(1, 4, 0), vec(1, 1, 0.8)));
    lights.push_back(Light(vec(2, 0.0, -0.2), vec(1, 1, 1)));
	
    material *red_dif = new material(MAT_DIFFUSE);
    red_dif->ambient_color = vec(1, 1, 1);
    
    material *red_metal = new material(MAT_METAL, 1.5);
    red_metal->k_mirror = 0.9;
    red_metal->ambient_color = vec(1, 0.9, 0.1);
    
    material *glass = new material(MAT_DIELECTRIC, 1.6);
    glass->k_reflect = 0.4;
    
    
    //objects.add( make_shared<Sphere>(vec(0, 1, -1), 0.5, red_metal->clone()) );
    //objects.add( make_shared<Sphere>(vec(-0.05, 0, -2), 0.1, red_dif->clone()) );
    
    //objects.add( make_shared<Sphere>(vec(0, 0, -0.6), 0.2, red_metal->clone()) );
    for (int i = 0; i < 10000; i++) {
		vec a(random_double() - 0.5, random_double(), -1);
		vec b = a + vec(0.5, random_double() * 0.5 - 0.25, 2) * random_double() + vec(0.2, 0.3, 0.05);
		vec c = a + vec(0.2, random_double() * 0.5 - 0.25, 2) * random_double() + vec(0.2, 0.1, 0.1);
		material *tri = new material(MAT_DIFFUSE);
		tri->ambient_color = vec(random_double(),random_double(),random_double());
		objects.add( make_shared<Triangle>(a, b, c, tri) );
	}
	
	//objects.add( make_shared<Circle>(vec(0, 0, -0.2), 0.1, vec(1, 0, -1), vec(0, 1, 0), red_dif->clone()) );
    
    //addcube(objects, vec(-3, -1, -2), red_dif);
    
    
    //objects.add( make_shared<InfinitePlane>(vec(0, -0.2, 0), vec(1, 0, 0), vec(0, 0, -1), red_dif->clone()) );
    
    objects.construct_BVH();
    
    
    
	std::thread t1(ray_caster, std::ref(img), 0, n,
            std::ref(objects), std::ref(lights), std::ref(main_camera), n, m);
    //std::thread t2(ray_caster, std::ref(img), n / 2, n,
    //        std::ref(objects), std::ref(main_camera), n, m, antialias);
    t1.join();
    //t2.join();
	

	img.save("hi.bmp");

    return 0;
}
