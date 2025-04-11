
#include <iostream>
#include <vector>
#include <algorithm>
#include <memory>
#include <thread>

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "bitmap_image.hpp"

using std::cout;
using std::vector;
using std::shared_ptr;
using std::make_shared;
using std::thread;


const float INF = 1e6;
const float EPS = 1e-6;



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

int out1 = 0;

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
	if (threads == 1) {
		std::string path("hi.bmp");
		img_all.save(path);
	} else {
		int mem_begin = n_start * (m_img * 3);
		int mem_end = n_end * (m_img * 3);
		memcpy(img_ptr->image + mem_begin, img_all.image + mem_begin, mem_end - mem_begin);
	}
}

int main() {
	vector<std::thread> pool;
	vector<Image*> pool_buf;
	int threads = 1;
	
	float fov_h = 120 * M_PI / 180;
	float fov_v = 100 * M_PI / 180;
	float dpi = 500;
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
