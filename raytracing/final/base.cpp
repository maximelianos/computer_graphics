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
#include <stdlib.h>
#include <string.h>

#include "bitmap_image.hpp"

using std::cout;
using std::vector;
using std::shared_ptr;
using std::make_shared;
using std::thread;



struct cmd_args {
	const char *out_path = "hi.bmp"; // The contents are const, not the pointer
	int scene = 1;
	int threads = 1;
};

void cmd_parse(int argc, char **argv, cmd_args &args) {
	for (int i = 1; i < argc; i++) { // argv begins with command name
		if (strcmp(argv[i], "-out") == 0) {
			args.out_path = argv[i + 1];
		} else if (strcmp(argv[i], "-scene") == 0) {
			args.scene = strtol(argv[i + 1], nullptr, 10);
		} else if (strcmp(argv[i], "-threads") == 0) {
			args.threads = strtol(argv[i + 1], nullptr, 10);
		}
	}
}


namespace block_scene_12 {
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



	void read_poly(HittableList &objects, const char* filename, vec offset, float *rotate_mat) {
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
		vec backz(0, 0, -0.1);
		for (int i = 0; i < c; i++) {
			fin >> A[0] >> A[1] >> A[2];
			fin >> B[0] >> B[1] >> B[2];
			fin >> C[0] >> C[1] >> C[2];
			
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

	void scene_1(HittableList &objects, vector<Light> &lights) {
		lights.push_back(Light(vec(3, 1, 0.3), vec(0.8, 0.4, 0.2)));
		
		material *red_dif = new material(MAT_DIFFUSE);
		red_dif->ambient_color = vec(1, 0.5, 1);
		red_dif->k_diffuse = 0.8;
		
		material *red_metal = new material(MAT_METAL, 1.5);
		red_metal->k_mirror = 0.7;
		red_metal->ambient_color = vec(1, 0, 0.3);
		
		vector< shared_ptr<Hittable> > regular;
		vector< shared_ptr<Hittable> > special;
		
		special.push_back( make_shared<Circle>(vec(0, -1, 0), 10, vec(1, 0, 0), vec(0, 0, -1), red_metal->clone()) );
		
		float mat_x[9];
		float mat_y[9];
		float mat_z[9];
		float phi;
		
		// Letter 5
		phi = (30 + 20 * 0 + 10) * M_PI / 180;
		rot_x(-90 * M_PI / 180, mat_x);
		rot_y(0 * M_PI / 180, mat_y);
		rot_z(-M_PI/2 + phi, mat_z);
		mult_mat(mat_x, mat_y);
		mult_mat(mat_x, mat_z);
		
		read_poly(objects, "letters/letter-5-list.in", vec(cos(phi) * 3, -0.9, -sin(phi) * 3), mat_x);
		
		// Letter 4
		phi = (30 + 20 * 1 + 10) * M_PI / 180;
		rot_x(-5 * M_PI / 180, mat_x);
		rot_y(-M_PI/2 + phi, mat_y);
		rot_z(-15 * M_PI / 180, mat_z);
		mult_mat(mat_x, mat_y);
		mult_mat(mat_x, mat_z);
		
		read_poly(objects, "letters/letter-4-list.in", vec(cos(phi) * 3, -0.8, -sin(phi) * 3), mat_x);
		
		// Letter 3
		phi = (30 + 20 * 2 + 10) * M_PI / 180;
		rot_x(-80 * M_PI / 180, mat_x);
		rot_y(0, mat_y);
		rot_z(-M_PI/2 + (phi + 30 * M_PI / 180), mat_z);
		mult_mat(mat_x, mat_y);
		mult_mat(mat_x, mat_z);
		
		read_poly(objects, "letters/letter-3-list.in", vec(cos(phi) * 1.5, -0.9, -sin(phi) * 1.5), mat_x);
		
		// Letter 2
		phi = (30 + 20 * 3 + 10) * M_PI / 180;
		rot_x(-90 * M_PI / 180, mat_x);
		rot_y(0, mat_y);
		rot_z(-M_PI/2 + (phi - 20 * M_PI / 180), mat_z);
		mult_mat(mat_x, mat_y);
		mult_mat(mat_x, mat_z);
		
		read_poly(objects, "letters/letter-2-list.in", vec(cos(phi) * 3, -0.9, -sin(phi) * 3), mat_x);
		
		// Letter 1
		phi = (30 + 20 * 4 + 10) * M_PI / 180;
		rot_x(0 * M_PI / 180, mat_x);
		rot_y(-M_PI/2 + phi, mat_y);
		rot_z(0, mat_z);
		mult_mat(mat_x, mat_y);
		mult_mat(mat_x, mat_z);
		
		read_poly(objects, "letters/letter-1-list.in", vec(cos(phi) * 3, -0.66, -sin(phi) * 3), mat_x);
		
		// Letter 0
		phi = (30 + 20 * 5 + 10) * M_PI / 180;
		rot_x(-90 * M_PI / 180, mat_x);
		rot_y(0, mat_y);
		rot_z(-M_PI/2 + phi, mat_z);
		mult_mat(mat_x, mat_y);
		mult_mat(mat_x, mat_z);
		
		read_poly(objects, "letters/letter-0-list.in", vec(cos(phi) * 3, -0.9, -sin(phi) * 3), mat_x);
		
		
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

	void main_wrapped(Image *img_ptr, int scene, int threads, int thr,
		float fov_h, float fov_v, int n_img, int m_img) {
		
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
		
		if (scene == 1) {
			EPS = 1e-5;
			bkg.load("panorama.bmp");
			main_camera = Camera(fov_h, fov_v, -25, vec(0.0, 1.0, 0));
			scene_1(objects, lights);
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
		int mem_begin = n_start * (m_img * 3);
		int mem_end = n_end * (m_img * 3);
		memcpy(img_ptr->image + mem_begin, img_all.image + mem_begin, mem_end - mem_begin);
	}
	
	void main(cmd_args &args) {
		srand(0);
		vector<std::thread> pool;
		vector<Image*> pool_buf;
		int threads = args.threads;
		
		float fov_h = 120 * M_PI / 180;
		float fov_v = 100 * M_PI / 180;
		float dpi = 220;
		int n_img = tan(fov_v / 2) * 2 * dpi;
		int m_img = tan(fov_h / 2) * 2 * dpi;
		
		Image img_all(n_img, m_img);
		for (int thr = 0; thr < threads; thr++) {
			pool_buf.push_back(new Image(n_img, m_img));
			
			pool.push_back(
			std::thread(main_wrapped, pool_buf[thr], args.scene, threads, thr, fov_h, fov_v, n_img, m_img) );
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
		
		std::string path(args.out_path);
		img_all.save(path);
	} 
}

namespace block_scene_3 {
	const float EPS = 1e-4;
	const float INF = 1e6;


	class vec3 {
		float e[3];

	public:
		vec3() {}
		vec3(float e0, float e1, float e2) { e[0] = e0; e[1] = e1; e[2] = e2; }
		inline float x() const { return e[0]; }
		inline float y() const { return e[1]; }
		inline float z() const { return e[2]; }
		inline float r() const { return e[0]; }
		inline float g() const { return e[1]; }
		inline float b() const { return e[2]; }

		inline const vec3& operator+() const { return *this; }
		inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
		inline float operator[](int i) const { return e[i]; }
		inline float& operator[](int i) { return e[i]; }

		inline vec3& operator+=(const vec3 &v2);
		inline vec3& operator-=(const vec3 &v2);
		inline vec3& operator*=(const vec3 &v2);
		inline vec3& operator/=(const vec3 &v2);
		inline vec3& operator*=(const float t);
		inline vec3& operator/=(const float t);

		inline float length() const { return sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]); }
		inline float squared_length() const { return e[0]*e[0] + e[1]*e[1] + e[2]*e[2]; }
		inline void make_unit_vector();

		friend inline vec3 operator+(const vec3 &v1, const vec3 &v2);
		friend inline vec3 operator-(const vec3 &v1, const vec3 &v2);
		friend inline vec3 operator*(const vec3 &v1, const vec3 &v2);
		friend inline vec3 operator/(const vec3 &v1, const vec3 &v2);
		friend inline vec3 operator*(float t, const vec3 &v);
		friend inline vec3 operator*(const vec3 &v, float t);
		friend inline vec3 operator/(const vec3 &v, float t);

		friend inline float dot(const vec3 &v1, const vec3 &v2);
		friend inline vec3 cross(const vec3 &v1, const vec3 &v2);

		friend inline std::istream& operator>>(std::istream &is, vec3 &t);
		friend inline std::ostream& operator<<(std::ostream &os, const vec3 &t);

	};

	inline std::istream& operator>>(std::istream &is, vec3 &t) {
		is >> t.e[0] >> t.e[1] >> t.e[2];
		return is;
	}

	inline std::ostream& operator<<(std::ostream &os, const vec3 &t) {
		os << t.e[0] << " " << t.e[1] << " " << t.e[2];
		return os;
	}

	inline void vec3::make_unit_vector() {
		float k = 1.0 / sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
		e[0] *= k; e[1] *= k; e[2] *= k;
	}

	inline vec3 operator+(const vec3 &v1, const vec3 &v2) {
		return vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
	}

	inline vec3 operator-(const vec3 &v1, const vec3 &v2) {
		return vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
	}

	inline vec3 operator*(const vec3 &v1, const vec3 &v2) {
		return vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
	}

	inline vec3 operator*(float t, const vec3 &v) {
		return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
	}

	inline vec3 operator*(const vec3 &v, float t) {
		return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
	}

	inline vec3 operator/(const vec3 &v1, const vec3 &v2) {
		return vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
	}

	inline vec3 operator/(const vec3 &v, float t) {
		return vec3(v.e[0]/t, v.e[1]/t, v.e[2]/t);
	}

	inline float dot(const vec3 &v1, const vec3 &v2) {
		return v1.e[0]*v2.e[0]
			 + v1.e[1]*v2.e[1]
			 + v1.e[2]*v2.e[2];
	}

	inline vec3 cross(const vec3 &v1, const vec3 &v2) {
		return vec3(v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1],
					v1.e[2] * v2.e[0] - v1.e[0] * v2.e[2],
					v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]);
	}

	inline vec3& vec3::operator+=(const vec3 &v) {
		e[0] += v.e[0];
		e[1] += v.e[1];
		e[2] += v.e[2];
		return *this;
	}

	inline vec3& vec3::operator-=(const vec3& v) {
		e[0] -= v.e[0];
		e[1] -= v.e[1];
		e[2] -= v.e[2];
		return *this;
	}

	inline vec3& vec3::operator*=(const vec3 &v) {
		e[0] *= v.e[0];
		e[1] *= v.e[1];
		e[2] *= v.e[2];
		return *this;
	}

	inline vec3& vec3::operator*=(const float t) {
		e[0] *= t;
		e[1] *= t;
		e[2] *= t;
		return *this;
	}

	inline vec3& vec3::operator/=(const vec3 &v) {
		e[0] /= v.e[0];
		e[1] /= v.e[1];
		e[2] /= v.e[2];
		return *this;
	}

	inline vec3& vec3::operator/=(const float t) {
		float k = 1.0/t;

		e[0] *= k;
		e[1] *= k;
		e[2] *= k;
		return *this;
	}

	inline vec3 unit_vector(vec3 v) {
		return v / v.length();
	}



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
		
		void set(int row, int col, vec3 color) {
			int idx = row * (n_cols * n_colors) + col * n_colors;
			image[idx] = int(color[0]);
			image[idx + 1] = int(color[1]);
			image[idx + 2] = int(color[2]);
		}
		
		void set(int idx, vec3 color) {
			image[idx] = color[0];
			image[idx + 1] = color[1];
			image[idx + 2] = color[2];
		}
		
		vec3 get(int row, int col) const {
			int idx = row * (n_cols * n_colors) + col * n_colors;
			return vec3(image[idx], image[idx + 1], image[idx + 2]);
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



	unsigned int module = 6700417;
	unsigned int mul_k = 41;
	unsigned int bias_c = 100001;
	unsigned int cu_x = 204048;
	float mxrand = module + 1.0;


	inline float random_double() {
		//cu_x = (cu_x * mul_k + bias_c) % module;
		//return cu_x / mxrand;
		return (float) (random() / (RAND_MAX + 1.0));
	}

	vec3 random_in_unit_sphere() {
		/*vec3 p(0, 0, 0);
		do {
			p = 2 * vec3(random_double(), random_double(), random_double()) - vec3(1,1,1);
		} while (p.squared_length() >= 1.0);*/

		/*float phi = random_double() * 2 * M_PI;
		float psi = (random_double() - 0.5) * M_PI;
		float r = random_double();
		vec3 p = vec3(cos(phi), sin(phi), sin(psi)) * r;*/

		float phi = random_double() * 2 * M_PI;
		float z = random_double() * 2 - 1;
		float r = sqrt(1 - z * z);
		vec3 p = vec3(r * cos(phi), r * sin(phi), z);

		/*float y = random_double() * 4;
		float x;
		if (y < 2) {
			y -= 1;
			x = sqrt(1 - y * y);
		} else {
			y -= 3;
			x = -sqrt(1 - y * y);
		}
		float z = random_double() * 2 - 1;
		float r = sqrt(1 - z * z);
		vec3 p = vec3(r * x, r * y, z);*/

		return p;
	}

	vec3 mult_v_mat(vec3 v, float *mat) {
		vec3 res;
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
		vec3 A;
		vec3 B;

	public:
		Ray() {}
		Ray(const vec3 &a, const vec3 &b) { A = a; B = b; }
		vec3 origin() const       { return A; }
		vec3 direction() const    { return B; }
		vec3 point_at_parameter(float t) const { return A + t*B; }
	};



	class Camera {
		vec3 origin;
		float h_fov, v_fov;
		vec3 lower_left_corner;
		vec3 horizontal;
		vec3 vertical;

	public:
		Camera(float horizontal_rad_fov=M_PI, float vertical_rad_fov=M_PI, float rx=0, vec3 pos=vec3(0, 0, 0)) {
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
			lower_left_corner = mult_v_mat(vec3(-half_h, -half_v, -1.0), mat_x);
			horizontal = vec3(half_h * 2, 0.0, 0.0);
			vertical = mult_v_mat(vec3(0.0, half_v * 2, 0.0), mat_x);
			origin = pos;
		}
		Ray get_ray(float u, float v) const {
			return Ray(origin, lower_left_corner + u*horizontal + v*vertical);
		}


	};



	class Material;
	struct hit_record {
		float t;
		vec3 p;
		vec3 normal;
		vec3 color = vec3(-1, 0, 0);
		shared_ptr<Material> mat_ptr;
	};

	// Abstract class for rendering any surface or volume
	class Hittable  {
	public:
		virtual bool hit(
			const Ray &r, float t_min, float t_max, hit_record &rec) const = 0;
	};

	class HittableList: public Hittable {
		vector< shared_ptr<Hittable> > l; // Without pointer it won't compile

	public:
		HittableList() {}
		HittableList(const vector< shared_ptr<Hittable> > &list) { l = list; }

		//vector< shared_ptr<Hittable> >& get_list() { return l; }

		void add(const shared_ptr<Hittable> &object) {
			l.push_back(object);
		}

		virtual bool hit(
			const Ray &r, float t_min, float t_max, hit_record& rec) const;
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
	}



	class Material {
	public:
		/*
		 * On contact with surface, produce a new scattered ray.
		 * attenuation defines how much each color channel will be
		 * attenuated.
		 * Returns false if there will be no scattered ray (input
		 * ray was absorbed).
		 */
		virtual bool scatter(
			const Ray &r_in, const hit_record &rec, vec3 &attenuation, Ray &scattered
		) const = 0;
		
		virtual vec3 radiance() const {
			return vec3(0, 0, 0);
		}
	};

	inline vec3 reflect(const vec3 &vec_in, const vec3 &normal) {
		return vec_in + 2 * normal * dot(vec_in, normal) * (-1);
	}

	class Lambertian : public Material {
		vec3 albedo;

	public:
		Lambertian(const vec3 &a) { albedo = a; }

		virtual bool scatter(
			const Ray &r_in, const hit_record &hit_rec, vec3 &attenuation, Ray &scattered
		) const {
			// attenuation and scattered are used as return values

			const vec3 normal = dot(r_in.direction(), hit_rec.normal) < 0 ?
					hit_rec.normal : -hit_rec.normal;
			vec3 scatter_direction = normal + random_in_unit_sphere();
			scattered = Ray(hit_rec.p, scatter_direction);
			attenuation = albedo;
			return true;
		}
	};

	class UniformLight : public Material {
		vec3 radiance_color;

	public:
		UniformLight(vec3 radiance_color) : radiance_color(radiance_color) {}

		virtual bool scatter(
			const Ray &r_in, const hit_record &hit_rec, vec3 &attenuation, Ray &scattered
		) const {
			// attenuation and scattered are used as return values
			return false;
		}
		
		virtual vec3 radiance() {
			return radiance_color;
		}
	};

	class Metal : public Material {
		vec3 a;
		float f; // f is in [0, 1]

	public:
		Metal(const vec3 &albedo, const float fuzziness) { a = albedo; f = fuzziness; }

		virtual bool scatter(
			const Ray &r_in, const hit_record &hit_rec, vec3 &attenuation, Ray &scattered
		) const {
			// attenuation and scattered are used as return values

			const vec3 &vec_in = r_in.direction();
			float dot_v_n = dot(r_in.direction(), hit_rec.normal);
			const vec3 &normal = dot_v_n < 0 ? hit_rec.normal : -hit_rec.normal;
			float norm_h = fabs(dot_v_n);

			vec3 scatter_direction = vec_in + 2 * normal * norm_h +
					random_in_unit_sphere() * norm_h * f;

			scattered = Ray(hit_rec.p, scatter_direction);
			attenuation = a;
			return true;
		}
	};

	class Dielectric : public Material {
		vec3 a;
		float f; // f is in [0, 1]
		float th;
		float th_inv;
		float th_sq;
		float th_inv_sq;

	public:
		Dielectric(const vec3 &albedo, const float fuzziness, const float theta) {
			a = albedo;
			f = fuzziness;
			th = theta;
			th_inv = 1.0 / theta;
			th_sq = theta * theta;
			th_inv_sq = 1.0 / (theta * theta);
		}

		virtual bool scatter(
			const Ray &r_in, const hit_record &hit_rec, vec3 &attenuation, Ray &scattered
		) const {
			// attenuation and scattered are used as return values

			const vec3 &vec_in = r_in.direction();
			float dot_v_n = dot(r_in.direction(), hit_rec.normal);

			vec3 normal;
			float theta;
			float theta_sq;
			if (dot_v_n < 0) { // Ray enters body under surface
				normal = hit_rec.normal;
				theta = th;
				theta_sq = th_sq;
			} else {
				normal = -hit_rec.normal;
				theta = th_inv;
				theta_sq = th_inv_sq;
			}
			float norm_h = fabs(dot_v_n);

			vec3 u = vec_in + normal * norm_h;

			float cos_a = norm_h / vec_in.length();
			float sin_a_sq = 1 - cos_a * cos_a;

			float r0 = (1 - theta) / (1 + theta); // Shlick formula
			r0 = r0 * r0;
			float reflect_probability = r0 + (1 - r0) * pow(1 - cos_a, 5);

			if (sin_a_sq >= theta_sq || random_double() < reflect_probability) {
				// Only reflect
				vec3 scatter_direction = vec_in + 2 * normal * norm_h +
						random_in_unit_sphere() * norm_h * f;
				scattered = Ray(hit_rec.p, scatter_direction);
			} else {
				// Refract through surface
				float sin_a = sqrt(sin_a_sq);
				vec3 refracted = -normal * sqrt(theta_sq - sin_a_sq) +
						unit_vector(u) * sin_a;
				refracted = refracted +
						random_in_unit_sphere() * dot(refracted, normal) * f;

				scattered = Ray(hit_rec.p, refracted);
			}
			attenuation = a;
			return true;
		}
	};



	class Sphere: public Hittable {
		vec3 C;
		float r;
		shared_ptr<Material> mat_ptr;

	public:
		Sphere() {}
		Sphere(
			const vec3 &center, const float radius, const shared_ptr<Material> material
		) { C = center; r = radius; mat_ptr = material; mat_ptr = material; }

		vec3 center() const 	{ return C; }
		float radius() const 	{ return r; }
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
					rec.normal = unit_vector((rec.p - C) / r);
					rec.mat_ptr = mat_ptr;
					return true;
				}

				float x2 = (-b + sqrt(d)) / a;
				if (t_min < x2 && x2 < t_max) {
					rec.t = x2;
					rec.p = ray.point_at_parameter(x2);
					rec.normal = unit_vector((rec.p - C) / r);
					rec.mat_ptr = mat_ptr;
					return true;
				}
			}

			return false;
		}
	};

	class InfinitePlane: public Hittable {
		vec3 p; // Origin point
		vec3 u, v; // Direction vectors
		vec3 normal;

		float a, b, c, d;
		shared_ptr<Material> mat_ptr;

	public:
		InfinitePlane() {}
		InfinitePlane(
			const vec3 &point, const vec3 &dir1, const vec3 &dir2, shared_ptr<Material> material
		) {
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

			normal = unit_vector(vec3(tx, -ty, -tz));

			v = cross(normal, u);
			mat_ptr = material;
		}

		bool hit(const Ray &ray, float t_min, float t_max, hit_record &rec) const {
			// M = p + t_1*u + t_2*v = q + t_3*s =>
			// t_1*u + t_2*v - t_3*s = q - p
			vec3 s = -ray.direction();
			vec3 origin = ray.origin() - p;
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

			vec3 f = ray.origin() + t * (-s) - p;
			float det_1 = f.x() * v.z() - v.x() - f.z();
			float det_2 = u.x() * f.z() - f.x() * u.z();
			float det_all2 = u.x() * v.z() - v.x() * u.z();
			float t1 = det_1 / det_all2;
			float t2 = det_2 / det_all2;

			if (t_min < t && t < t_max) {
				rec.t = t;
				rec.p = ray.point_at_parameter(t);
				rec.normal = normal;
				rec.color = vec3(1, 1, 1) * (((unsigned(t1 * 50) / 10) % 2 + (unsigned(t2 * 50) / 10) % 2) % 2);
				rec.mat_ptr = mat_ptr;
				return true;
			}

			return false;
		}
	};

	class Triangle: public Hittable {
		vec3 p; // Origin point
		vec3 u, v; // Direction vectors
		vec3 normal;

		float a, b, c, d;
		shared_ptr<Material> mat_ptr;
		bool proj_x, proj_y, proj_z;
		
		Image tex;
		vec3 tex_p;
		vec3 tex_u, tex_v;

	public:
		vec3 force_color = vec3(-1000, 0, 0);

		Triangle() {}
		Triangle(
			const vec3 &A, const vec3 &B, const vec3 &C, shared_ptr<Material> material
		) {
			// a, b, c - in clockwise order to camera
			
			vec3 point = B;
			vec3 dir1 = A - B;
			vec3 dir2 = C - B;
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

			normal = unit_vector(vec3(tx, -ty, -tz));

			//v = cross(normal, u);
			mat_ptr = material;
			
			vec3 test_u = vec3(0, u.y(), u.z());
			vec3 test_v = vec3(0, v.y(), v.z());
			
			proj_x = !(test_u.length() < EPS || test_v.length() < EPS ||
					(fabs(dot(unit_vector(test_u), unit_vector(test_v))) > 1 - EPS));
			
			test_u = vec3(u.x(), 0, u.z());
			test_v = vec3(v.x(), 0, v.z());
			
			proj_y = !(test_u.length() < EPS || test_v.length() < EPS ||
					(fabs(dot(unit_vector(test_u), unit_vector(test_v))) > 1 - EPS));
			
			test_u = vec3(u.x(), u.y(), 0);
			test_v = vec3(v.x(), v.y(), 0);
			
			proj_z = !(test_u.length() < EPS || test_v.length() < EPS ||
					(fabs(dot(unit_vector(test_u), unit_vector(test_v))) > 1 - EPS));
		}
		
		void set_texture(const char *img_filename, vec3 A, vec3 B, vec3 C) {
			tex.load(img_filename);
			tex_p = B;
			tex_u = A - B;
			tex_v = C - B;
		}

		bool hit(const Ray &ray, float t_min, float t_max, hit_record &rec) const {
			vec3 ray_pos = ray.origin() - p;
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
			vec3 s = -ray.direction();
			vec3 origin = ray.origin() - p;
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

			vec3 f = ray.point_at_parameter(t) - p;
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
			float t1 = det_1 / det_all2; // u component
			float t2 = det_2 / det_all2; // v component

			if (t1 > 0 && t2 > 0 && t1 < 1 - t2 &&
					t_min < t && t < t_max) {
				rec.t = t;
				rec.p = ray.point_at_parameter(t);
				rec.normal = normal;
				rec.mat_ptr = mat_ptr;
				if (force_color[0] >= 0) {
					rec.color = force_color;
				}
				
				//cout << t1 << " " << t2 << "\n";
				//cout << tex_u << " " << tex_v << "\n";
				vec3 tex_coor = tex_p + t1 * tex_u + t2 * tex_v;
				//cout << tex_coor << "\n";
				rec.color = tex.get(tex_coor[1], tex_coor[0]) / 255;
				return true;
			}

			return false;
		}
	};

	vec3 bkg_color(const Ray &r) {
		vec3 unit_direction = unit_vector(r.direction());
		float t = 0.5*(unit_direction.y() + 1.0);
		return (1-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
	}

	int reflections = 0;

	vec3 color(const Ray &ray, const HittableList &objects, int recursion) {
		if (recursion > 30) {
			//printf("Exceeded\n");
			return vec3(0, 0, 0);
		}

		vec3 col;
		hit_record hit_rec;

		if (objects.hit(ray, 0.0001, INF, hit_rec)) {
			reflections++;
		
		
			vec3 attenuation;
			Ray scattered;
			vec3 radiance = hit_rec.mat_ptr->radiance();
			if (hit_rec.mat_ptr->scatter(ray, hit_rec, attenuation, scattered)) {
				col = radiance + color(scattered, objects, recursion + 1) * attenuation;
				
			} else { // Ray absorbed :(
				col = radiance;
			}
			
			if (hit_rec.color[0] >= 0) { // Force a specific color
				col *= hit_rec.color;
			}
		
		} else {
			col = bkg_color(ray);
		}
		return col;
	}



	void ray_caster(
		Image &img, int row_begin, int row_end,
		HittableList &objects, Camera &main_camera, int n, int m, int antialias)
	{
		for (int i = row_begin; i < row_end; i++) {
			for (int j = 0; j < m; j++) {
				vec3 col(0, 0, 0);
				for (int it = 0; it < antialias; it++) {
					float v = (float(n - i) + random_double()) / float(n);
					float u = (float(j) + random_double()) / float(m);
					Ray r = main_camera.get_ray(u, v);
					col += color(r, objects, 0);
				}
				col /= (float) antialias;
				
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
					printf("Color out of [0.0, 1.0] range! Exiting...\n");
					exit(0);
				}

				img(i, j, 0) = sqrt(col[0]) * 255;
				img(i, j, 1) = sqrt(col[1]) * 255;
				img(i, j, 2) = sqrt(col[2]) * 255;
			}
		}
	}

	void scene_3(HittableList &objects) {
		objects.add( make_shared<Sphere>(vec3(0, -100, 0), 99.5, make_shared<Metal>(vec3(0.9, 0.9, 0.1), 0)) );
		objects.add( make_shared<Sphere>(vec3(0, 0, -2.2), 0.4, make_shared<Lambertian>(vec3(0.9, 0.1, 0.1))) );
		objects.add( make_shared<Sphere>(vec3(-1, 0, -1), 0.4, make_shared<Metal>(vec3(0.8, 0.85, 0.9), 0.2)) );
		
		objects.add( make_shared<Sphere>(vec3(1.2, 0, -1), 0.3, make_shared<Metal>(vec3(1, 0.1, 1), 0)) );
		
		objects.add( make_shared<Sphere>(vec3(0, 0.2, -1), 0.6, make_shared<Dielectric>(vec3(1, 1, 1), 0., 1.6)) );
		objects.add( make_shared<Sphere>(vec3(-0.3, 0.2, -1), 0.2, make_shared<Dielectric>(vec3(1, 1, 1), 0., 1.4)) );
		objects.add( make_shared<Sphere>(vec3(0.3, 0.2, -1), 0.2, make_shared<Dielectric>(vec3(1, 1, 1), 0., 1.4)) );
		
		objects.add( make_shared<Sphere>(vec3(0.3, -0.1, -0.3), 0.1, make_shared<Dielectric>(vec3(1, 1, 1), 0., 1.6)) );
		
		Image tex_test;
		tex_test.load("portrait.bmp");
		float h = tex_test.n_rows - 1;
		float w = tex_test.n_cols - 1;
		
		float k = 180;
		vec3 shf = vec3(-w / k / 2, -h / k / 2, -5);
		vec3 a = vec3(w / k, 0, 0) + shf;
		vec3 b = vec3(0, 0, 0) + shf;
		vec3 c = vec3(0, h / k, 0) + shf;
		shared_ptr<Triangle> poster = make_shared<Triangle>(
			a, b, c, make_shared<Lambertian>(vec3(1.0, 1.0, 1.0))
		);
		
		poster->set_texture("portrait.bmp", vec3(w, h, 0), vec3(0, h, 0), vec3(0, 0, 0));
		objects.add(poster);
		
		a = vec3(w / k, 0, 0) + shf;
		b = vec3(w / k, h / k, 0) + shf;
		c = vec3(0, h / k, 0) + shf;
		shared_ptr<Triangle> poster2 = make_shared<Triangle>(
			a, b, c, make_shared<Lambertian>(vec3(1.0, 1.0, 1.0))
		);
		
		poster2->set_texture("portrait.bmp", vec3(w, h, 0), vec3(w, 0, 0), vec3(0, 0, 0));
		objects.add(poster2);
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
		Camera main_camera(fov_h, fov_v);
		HittableList objects;
		
		int antialias = 12;
		scene_3(objects);
		
		// Run scene
		ray_caster(img_all, n_start, n_end, objects, main_camera, n_img, m_img, antialias);
		
		// Save results
		int mem_begin = n_start * (m_img * 3);
		int mem_end = n_end * (m_img * 3);
		memcpy(img_ptr->image + mem_begin, img_all.image + mem_begin, mem_end - mem_begin);
	}
	
	void main(cmd_args &args) {
		srand(0);
		vector<std::thread> pool;
		vector<Image*> pool_buf;
		int threads = args.threads;
		
		float fov_h = 120 * M_PI / 180;
		float fov_v = 100 * M_PI / 180;
		float dpi = 400;
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
		
		std::string path(args.out_path);
		img_all.save(path);
	}
}





int main(int argc, char **argv) {
	cmd_args args;
	cmd_parse(argc, argv, args);
	
	if (args.scene < 3) {
		block_scene_12::main(args);
	} else {
		block_scene_3::main(args);
	}
	
    return 0;
}
