use std::{ops, f32::consts::PI, collections::VecDeque, time::Instant};

use grid::Grid;
use image::GrayImage;
mod loadsdf;

#[derive(Default, Copy, Clone, Debug, PartialEq)]
struct Point {
    x: f32,
    y: f32
}

impl Point {
    fn new(x: f32, y: f32) -> Point {
        return Point{x: x, y: y};
    }

    fn len(&self) -> f32 {
        return (self.x * self.x + self.y * self.y).sqrt();
    }

    fn dot(&self, b: Point) -> f32 {
        return self.x * b.x + self.y * b.y;
    }

    fn cross(&self, b: Point) -> f32 {
        return self.x * b.y - b.x * self.y;
    }

    fn proj(&self, b: Point) -> f32 {
        return self.dot(b) / self.len();
    }

    fn rotate(&self, alpha: f32) -> Point {
        return Point::new(
            self.x * alpha.cos() - self.y * alpha.sin(),
            self.x * alpha.sin() + self.y * alpha.cos());
    }
}

impl ops::Add<Point> for Point {
    type Output = Point;

    fn add(self, _rhs: Point) -> Point {
        //println!("> Point.add(Point) was called");
        return Point {x: self.x + _rhs.x, y: self.y + _rhs.y};
    }
}

impl ops::Mul<f32> for Point {
    type Output = Point;

    fn mul(self, _rhs: f32) -> Point {
        return Point {x: self.x * _rhs, y: self.y * _rhs};
    }
}

impl ops::Neg for Point {
    type Output = Self;

    fn neg(self) -> Point {
        //println!("> Point.neg() was called");
        return Point{x: -self.x, y: -self.y};
    }
}

impl ops::Sub<Point> for Point {
    type Output = Point;

    fn sub(self, _rhs: Point) -> Point {
        //println!("> Point.sub(Point) was called");
        return Point {x: self.x - _rhs.x, y: self.y - _rhs.y};
    }
}

struct ShapeTransform {
    scale: f32,
    rotate: f32, // in radians
    translate: Point
}

impl Default for ShapeTransform {
    fn default() -> Self { Self{scale: 1., rotate: 0., translate: Point::new(0., 0.)} }
}

impl ShapeTransform {
    fn apply(&self, p: Point) -> Point {
        // apply same transform to translation vector
        let translate = self.translate; //(self.translate * (1./self.scale)).rotate(-self.rotate);
        // transform in reverse order
        let mut q = p - translate;
        q = q.rotate(-self.rotate);
        q = q * (1./self.scale);
        return q;
    }
}

#[derive(Default, Copy, Clone, Debug, PartialEq)]
enum ShapeType {
    #[default]
    Circle,
    
    Rectangle,
    Triangle,
    LineSegment,
    Bitmap
}

#[derive(Default)]
struct Shape {
    stype: ShapeType,
    // circle requires C
    r: f32,
    // rect
    w: f32,
    h: f32,
    // tri
    A: Point,
    B: Point,
    C: Point,
    // segment requires A, B
    input_bitmap: GrayImage,
    bitmap: GrayImage, // can be None or GrayImage
    tosdf_scale: f32, // less makes smoother edges in binary image converted to SDF, doesn't affect absolute SDF values
    field_scale: f32, // for bitmaps: d *= field_scale
    color: ShapeColor,
    gradient_color: ShapeColor,
    gradient_p1: Point,
    gradient_p2: Point,

    local_transform: ShapeTransform,
    transform: ShapeTransform,
    layer: usize,
    no_smin: bool
}

#[derive(Default, Copy, Clone, Debug, PartialEq)]
struct ShapeColor {
    r: f32,
    g: f32,
    b: f32,
    layer: i32,
}

impl ShapeColor {
    fn new(r: i32, g: i32, b: i32) -> ShapeColor {
        ShapeColor { r: r as f32 / 255., g: g as f32 / 255., b: b as f32 / 255., layer: 0 }
    }
}

impl ops::Add<ShapeColor> for ShapeColor {
    type Output = ShapeColor;

    fn add(self, _rhs: ShapeColor) -> ShapeColor {
        //println!("> Point.add(Point) was called");
        return ShapeColor {
            r: self.r + _rhs.r,
            g: self.g + _rhs.g,
            b: self.b + _rhs.b,
            layer: self.layer };
    }
}

impl ops::Mul<f32> for ShapeColor {
    type Output = ShapeColor;

    fn mul(self, rhs: f32) -> Self::Output {
        return ShapeColor {
            r: self.r * rhs,
            g: self.g * rhs,
            b: self.b * rhs,
            layer: self.layer };
    }
}

impl Shape {
    fn build(&mut self) {
        use ShapeType::*;
        match self.stype {
            Circle => (),
            Rectangle => (),
            Triangle => {
                // order points in clockwise manner
                let a = self.B - self.A;
                let b = self.C - self.A;
                //println!("> Cross product: {}", a.cross(b));
                if a.cross(b) > 0. {
                    let t = self.B;
                    self.B = self.C;
                    self.C = t;
                }

            },
            LineSegment => (),
            Bitmap => ()
        }
    }

    fn distance_pre(&self, point: Point) -> f32 {
        use ShapeType::*;
        // transform is inverse: global space -> local space -> object space
        let p = self.local_transform.apply(self.transform.apply(point));
        match self.stype {
            Circle => {
                return (p - self.C).len() - self.r;
            },
            Rectangle => {
                let w2 = self.w / 2.0;
                let h2 = self.h / 2.0;
                if -w2 < p.x && p.x < w2 {
                    return p.y.abs() - h2;
                } else if -h2 < p.y && p.y < h2 {
                    return p.x.abs() - w2;
                } else {
                    return Point::new(p.x.abs() - w2, p.y.abs() - h2).len();
                }
            },
            Triangle => {
                let a = self.B - self.A;
                let b = self.C - self.B;
                let c = self.A - self.C;
                let pa = p - self.A;
                let pb = p - self.B;
                let pc = p - self.C;
                // on which side of triangle is our point
                let ca = a.cross(pa);
                let cb = b.cross(pb);
                let cc = c.cross(pc);
                let pja = a.proj(pa);
                let pjb = b.proj(pb);
                let pjc = c.proj(pc);
                // case A
                if ca >= 0. && 0. <= pja && pja <= a.len() {
                    return ca / a.len();
                } else if cb >= 0. && 0. <= pjb && pjb <= b.len() {
                    return cb / b.len();
                } else if cc >= 0. && 0. <= pjc && pjc <= c.len() {
                    return cc / c.len();
                // case B
                } else if cb > 0. && pjb < 0. || ca > 0. && !(pja < 0.)  {
                    // a-b side boundary
                    return pb.len();
                } else if cc > 0. && pjc < 0. || cb > 0. && !(pjb < 0.)  {
                    // b-c side boundary
                    return pc.len();
                } else if ca > 0. && pja < 0. || cc > 0. && !(pjc < 0.)  {
                    // c-a side boundary
                    return pa.len();
                } else {
                    // inside triangle. cross product is negative thus max is needed
                    return f32::max(ca / a.len(), 
                        f32::max(cb / b.len(), cc / c.len()));
                }
            },
            LineSegment => {
                let a: Point = self.B - self.A;
                let pa: Point = p - self.A;
                let proj: f32 = a.proj(pa);
                if 0. <= proj && proj <= a.len() {
                    return a.cross(pa).abs() / a.len();
                } else {
                    return f32::min(pa.len(), (p - self.B).len());
                }
            },
            Bitmap => {
                let (w, h): (u32, u32) = self.bitmap.dimensions();
                let q = Point::new(
                    p.x * w as f32, // TODO: think about scaling bitmap
                    p.y * h as f32
                );
                
                // limits can be inaccurate, there are checks after this
                // if q.x < -0. || q.x > w as f32 - 1. || q.y < -0. || q.y > h as f32 - 1. {
                //     return 1000.;
                // }

                let scale: f32;
                if self.field_scale > 0. {
                    scale = self.field_scale;
                } else {
                    scale = 1.;
                }
                let dist_large: f32 = 1000.;

                let x = q.x.floor() as i32;
                let y = q.y.floor() as i32;

                let get_sdf = |px: i32, py: i32| -> f32 {
                    if 0 <= px && px < w as i32 && 0 <= py && py < h as i32 {
                        0.5 - self.bitmap.get_pixel(px as u32, py as u32).0[0] as f32 / 255. // TODO: very ugly, please fix
                    } else {
                        dist_large
                    }
                };
                let v1 = get_sdf(x, y);
                let v2 = get_sdf(x, y+1);
                let v3 = get_sdf(x+1, y+1);
                let v4 = get_sdf(x+1, y);
                
                let kx = q.x - x as f32;
                let ky = q.y - y as f32;

                let k1 = (1.-kx)*(1.-ky);
                let k2 = (1.-kx)*(ky);
                let k3 = (kx)*(ky);
                let k4 = (kx)*(1.-ky);
                return (k1*v1 + k2*v2 + k3*v3 + k4*v4)/4.*scale; // TODO: this is not true distance, please think about it
            }
        }
    }

    fn distance(&self, point: Point) -> f32 {
        // in case of scaling, we must multiply the sdf by scaling factor
        return self.distance_pre(point) * self.local_transform.scale * self.transform.scale;
    }

    fn gradient(&self, point: Point) -> ShapeColor {
        let p = self.local_transform.apply(self.transform.apply(point));
        let gp = self.gradient_p2 - self.gradient_p1;
        let mut k = 1. - gp.proj(p - self.gradient_p1) / gp.len();
        k = k.max(0.).min(1.);
        return self.color * k + self.gradient_color * (1. - k);
    }

    fn bitmap_to_sdf(&mut self, sdf_resolution: f32, sdf_padding: f32) {
        // sdf_resolution is a small number inside [0, 1], makes the SDF matrix small
        // sdf_padding is fraction of padding on all sides of image, to compute accurate SDF
        println!("Convert binary image to SDF");
        let (w, h): (u32, u32) = self.input_bitmap.dimensions();
        
        let w = (w as f32 * (1.+sdf_padding*2.)) as u32;
        let h = (h as f32 * (1.+sdf_padding*2.)) as u32;
        let mut inp_bitmap = image::ImageBuffer::new(w, h);
        let y_off = (h as f32 * sdf_padding / 2.) as i64;
        let x_off = (w as f32 * sdf_padding / 2.) as i64;
        image::imageops::overlay(&mut inp_bitmap, &self.input_bitmap, x_off, y_off);
        self.input_bitmap = inp_bitmap;
        

        println!("Input dimensions: {w}x{h}");
        let outh = (h as f32 * sdf_resolution) as u32;
        let outw = (w as f32 * sdf_resolution) as u32;
        println!("SDF dimensions: {outw}x{outh}");

        let get_pixel = |px: u32, py: u32| -> f32 {
            self.input_bitmap.get_pixel(px, py).0[0] as f32 / 255. // TODO: very ugly, please fix
        };

        let is_inside = |px: u32, py: u32| -> bool {
            get_pixel(px, py) > 0.5
        };

        let mut visited: Grid<u8> = Grid::new(h as usize, w as usize);
        let mut parent: Grid<(usize, usize)> = Grid::new(h as usize, w as usize);
        let mut result: Grid<f32> = Grid::new(h as usize, w as usize);
        let mut deque: VecDeque<(usize, usize)> = VecDeque::new();

        for y in 0..h {
            for x in 0..w {
                if is_inside(x, y) {
                    visited[y as usize][x as usize] = 1;
                    deque.push_back((x as usize, y as usize));
                }
            }
        }
        while !deque.is_empty() {
            let (x, y) = deque.pop_front().unwrap();
            // ************************** BFS
            let px = x as i32;
            let py = y as i32;
            let ux = x;
            let uy = y;
            if visited[uy][ux] == 2 {
                continue;
            }
            
            // println!("Pix at x={ux} y={uy}: {pix}");
            let mut min_dist: f32 = 1000.;
            let mut min_parent: (usize, usize) = (0, 0);
            if is_inside(ux as u32, uy as u32) {
                result[uy][ux] = -1.;
                parent[uy][ux] = (ux, uy);
                min_dist = -1.;
            }
            let cur_point = Point::new(ux as f32, uy as f32);
            for i in -2..3 {
                for j in -2..3 {
                    let nx = px + j;
                    let ny = py + i;
                    if 0 <= nx && nx < w as i32 && 0 <= ny && ny < h as i32 {
                        if visited[ny as usize][nx as usize] == 0 {
                            if i==0&&(j==-1||j==1) || j==0&&(i==-1||i==1) {
                                visited[ny as usize][nx as usize] = 1;
                                deque.push_back((nx as usize, ny as usize));
                            }
                        } else if visited[ny as usize][nx as usize] == 2 {
                            let (pnx, pny) = parent[ny as usize][nx as usize];
                            let par_point = Point::new(pnx as f32, pny as f32);
                            if (par_point - cur_point).len() < min_dist {
                                min_dist = (par_point - cur_point).len();
                                min_parent = (pnx, pny);
                                // println!("Distance between {:?} {:?} = {:?}", cur_point, par_point, min_dist);
                            }
                        }
                    }
                }
            }

            if !is_inside(ux as u32, uy as u32) {
                result[uy][ux] = min_dist;
                parent[uy][ux] = min_parent;
            }

            visited[uy][ux] = 2;
        }

        // ******************* Now propagate inward

        for y in 0..h {
            for x in 0..w {
                visited[y as usize][x as usize] = 0;
                if !is_inside(x, y) {
                    visited[y as usize][x as usize] = 1;
                    deque.push_back((x as usize, y as usize));
                }
            }
        }

        while !deque.is_empty() {
            let (x, y) = deque.pop_front().unwrap();
            // ************************** BFS
            let px = x as i32;
            let py = y as i32;
            let ux = x;
            let uy = y;
            if visited[uy][ux] == 2 {
                continue;
            }
            
            let mut min_dist: f32 = 1000.;
            let mut min_parent: (usize, usize) = (0, 0);
            if !is_inside(ux as u32, uy as u32) {
                parent[uy][ux] = (ux, uy);
            }
            let cur_point = Point::new(ux as f32, uy as f32);
            for i in -2..3 {
                for j in -2..3 {
                    let nx = px + j;
                    let ny = py + i;
                    if 0 <= nx && nx < w as i32 && 0 <= ny && ny < h as i32 {
                        if visited[ny as usize][nx as usize] == 0 {
                            if i==0&&(j==-1||j==1) || j==0&&(i==-1||i==1) {
                                visited[ny as usize][nx as usize] = 1;
                                deque.push_back((nx as usize, ny as usize));
                            }
                        } else if visited[ny as usize][nx as usize] == 2 {
                            let (pnx, pny) = parent[ny as usize][nx as usize];
                            let par_point = Point::new(pnx as f32, pny as f32);
                            if (par_point - cur_point).len() < min_dist {
                                min_dist = (par_point - cur_point).len();
                                min_parent = (pnx, pny);
                                // println!("Distance between {:?} {:?} = {:?}", cur_point, par_point, min_dist);
                            }
                        }
                    }
                }
            }

            let small_distance: f32 = 1.;
            if is_inside(ux as u32, uy as u32) {
                result[uy][ux] = -min_dist + small_distance; // negative, because inside of shape
                parent[uy][ux] = min_parent;
            }

            visited[uy][ux] = 2;
        }

        let sdf_scale = self.tosdf_scale;

        let mut imgbuf = image::ImageBuffer::new(outw, outh);
        for y in 0..outh {
            for x in 0..outw {
                let y_input = (y as f32 / sdf_resolution) as u32;
                let x_input = (x as f32 / sdf_resolution) as f32;
                let val = result[y_input as usize][x_input as usize];
                // println!("Result at x={x} y={y}: {val:.3} parent={:?}", parent[y as usize][x as usize]);
                let res: f32;
                res = (0.5 + -val * sdf_scale).max(0.).min(1.);

                let pixel = imgbuf.get_pixel_mut(x as u32, y as u32);
                *pixel = image::Luma([(res * 255.) as u8]);
            }
        }

        // imgbuf.save("tosdf.png").unwrap();

        self.bitmap = imgbuf;
    }
}

fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = (1. - (a - b).abs() / k).max(0.);
    let m = h * h * h / 2.;
    let s = m * k / 3.;
    return a.min(b) - s;
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum SceneType {
    All,
    Letters,
    Bird,
    Flame,
    Fox,
    Shapes,
    Art
}

fn main_letters(scene: SceneType, save_path: &str) {
    println!("\n=== Scene {:?}", scene);
    let mut start_time = Instant::now();

    let yellow = ShapeColor::new(255, 220, 3);
    let orange = ShapeColor::new(251, 133, 0);
    let cyan = ShapeColor::new(0, 128, 128);
    let pink = ShapeColor::new(255, 0, 102);
    let red = ShapeColor::new(204, 0, 0);
    let sky = ShapeColor::new(142, 202, 230);
    let azul = ShapeColor::new(0, 102, 230);
    let grey = ShapeColor::new(40, 42, 42);
    let leaf = ShapeColor::new(0, 153, 0);
    let grass = ShapeColor::new(128, 255, 128);
    let fur = ShapeColor::new(230, 230, 230);

    let main_aliasing_scale = 0.001;
    let main_border_scale = 0.004;
    let letter_aliasing_scale = 0.025;
    let letter_border_scale = 0.04;

    let scale = 0.08;

    let shapes_s = 1.;
    let shapes_coors = 1.5*shapes_s; //1.5;
    let shapes_scale = scale*shapes_s; //scale;
    let shapes_pos = Point::new(-scale*shapes_coors*0.5, -scale*shapes_coors*0.5);
    
    let mut s_circ1 = Shape {
        stype: ShapeType::Circle,
        r: 1.,
        transform: ShapeTransform {
            scale: shapes_scale, 
            rotate: 0.,
            translate: Point::new(scale*shapes_coors*0., scale*shapes_coors*0.) + shapes_pos
        },
        color: yellow,
        ..Default::default()
    };
    s_circ1.build();

    let mut s_circ2 = Shape {
        stype: ShapeType::Circle,
        r: 1.,
        transform: ShapeTransform {
            scale: shapes_scale, 
            rotate: 0.,
            translate: Point::new(scale*shapes_coors*1., scale*shapes_coors*0.) + shapes_pos
        },
        color: orange,
        ..Default::default()
    };
    s_circ2.build();

    let mut s_circ3 = Shape {
        stype: ShapeType::Circle,
        r: 1.,
        transform: ShapeTransform {
            scale: shapes_scale, 
            rotate: 0.,
            translate: Point::new(scale*shapes_coors*0., scale*shapes_coors*1.) + shapes_pos
        },
        color: cyan,
        ..Default::default()
    };
    s_circ3.build();

    let mut s_circ4 = Shape {
        stype: ShapeType::Circle,
        r: 1.,
        transform: ShapeTransform {
            scale: shapes_scale, 
            rotate: 0.,
            translate: Point::new(scale*shapes_coors*1., scale*shapes_coors*1.) + shapes_pos
        },
        color: pink,
        ..Default::default()
    };
    s_circ4.build();

    let mut s_tri = Shape {
        stype: ShapeType::Triangle,
        A: Point::new(1., 0.),
        B: Point::new(0., 0.),
        C: Point::new(0., -3.),
        transform: ShapeTransform {
            scale: shapes_scale, 
            rotate: 0.,
            translate: Point::new(shapes_scale*-1., 0.) + shapes_pos
        },
        ..Default::default()
    };
    s_tri.build();

    let mut s_tri2 = Shape {
        stype: ShapeType::Triangle,
        A: Point::new(-1., 0.),
        B: Point::new(0., 0.),
        C: Point::new(0., 3.),
        transform: ShapeTransform {
            scale: shapes_scale,
            rotate: 0.,
            translate: Point::new(scale*shapes_coors*1. + shapes_scale*1., scale*shapes_coors*1.) + shapes_pos
        },
        ..Default::default()
    };
    s_tri2.build();

    let mut s_tri3 = Shape {
        stype: ShapeType::Triangle,
        A: Point::new(0., -1.),
        B: Point::new(0., 0.),
        C: Point::new(-3., 0.),
        transform: ShapeTransform {
            scale: shapes_scale,
            rotate: 0.,
            translate: Point::new(0., scale*shapes_coors*1. + shapes_scale*1.) + shapes_pos
        },
        ..Default::default()
    };
    s_tri3.build();

    let mut s_tri4 = Shape {
        stype: ShapeType::Triangle,
        A: Point::new(3., -1.),
        B: Point::new(0., -1.),
        C: Point::new(0., 0.),
        transform: ShapeTransform {
            scale: shapes_scale,
            rotate: 0.,
            translate: Point::new(scale*shapes_coors*1., 0.) + shapes_pos
        },
        ..Default::default()
    };
    s_tri4.build();

    // ******************************** LETTERS

    let mut letterm = Shape {
        stype: ShapeType::Bitmap,
        bitmap: loadsdf::loadsdf("resources/m_sdf2.png"),
        local_transform: ShapeTransform {
            scale: 1., 
            rotate: 0., 
            translate: Point::new(-0.5, -0.5)
        },
        transform: ShapeTransform {
            scale: scale*8., 
            rotate: 0., 
            translate: Point::new(-scale*1.5, -scale*1.5)
        },
        color: orange,
        layer: 1,
        ..Default::default()
    };
    letterm.build();

    let mut letterv = Shape {
        stype: ShapeType::Bitmap,
        bitmap: loadsdf::loadsdf("resources/v_sdf2.png"),
        local_transform: ShapeTransform {
            scale: 1., 
            rotate: 0., 
            translate: Point::new(-0.5, -0.5)
        },
        transform: ShapeTransform {
            scale: scale*8., 
            rotate: 0., 
            translate: Point::new(scale*2., scale*1.5)
        },
        color: leaf,
        layer: 1,
        ..Default::default()
    };
    letterv.build();

    // ************************************************ FOX

    let fox_pos: Point;
    let fox_scale: f32;
    match scene {
        SceneType::All => {
            fox_pos = Point::new(-scale*3.7, scale*5.5);
            fox_scale = 0.5;
        },
        _ => {
            fox_pos = Point::new(-scale*0., scale*3.);
            fox_scale = 1.;
        }
    }

    let mut tri1 = Shape {
        stype: ShapeType::Triangle,
        A: Point::new(1., 0.),
        B: Point::new(0., 0.),
        C: Point::new(0., -1.),
        local_transform: ShapeTransform { 
            scale: 1., 
            rotate: 0., 
            translate: Point::new(-2., -3.)
        },
        transform: ShapeTransform {
            scale: scale*fox_scale,
            rotate: 0.,
            translate: fox_pos
        },
        color: fur,
        no_smin: true,
        ..Default::default()
    };
    tri1.build();

    let mut rect2 = Shape {
        stype: ShapeType::Rectangle,
        w: 1.,
        h: 2.,
        local_transform: ShapeTransform { 
            scale: 1., 
            rotate: 0., 
            translate: Point::new(-1.5, -2.)
        },
        transform: ShapeTransform {
            scale: scale*fox_scale,
            rotate: 0.,
            translate: fox_pos
        },
        color: orange,
        no_smin: true,
        ..Default::default()
    };
    rect2.build();

    let mut tri3 = Shape {
        stype: ShapeType::Triangle,
        A: Point::new(-1., 0.),
        B: Point::new(0., 0.),
        C: Point::new(0., 1.),
        local_transform: ShapeTransform { 
            scale: 1., 
            rotate: 0., 
            translate: Point::new(-1., -1.)
        },
        transform: ShapeTransform {
            scale: scale*fox_scale,
            rotate: 0.,
            translate: fox_pos
        },
        color: grey,
        no_smin: true,
        ..Default::default()
    };
    tri3.build();

    let mut tri4 = Shape {
        stype: ShapeType::Triangle,
        A: Point::new(-3., 0.),
        B: Point::new(0., 0.),
        C: Point::new(0., -3.),
        local_transform: ShapeTransform { 
            scale: 1., 
            rotate: 0., 
            translate: Point::new(2., 0.)
        },
        transform: ShapeTransform {
            scale: scale*fox_scale,
            rotate: 0.,
            translate: fox_pos
        },
        color: orange,
        no_smin: true,
        ..Default::default()
    };
    tri4.build();

    let mut tri5 = Shape {
        stype: ShapeType::Triangle,
        A: Point::new(2., 0.),
        B: Point::new(0., 0.),
        C: Point::new(0., 2.),
        local_transform: ShapeTransform { 
            scale: 1.,
            rotate: 0.,
            translate: Point::new(0., -3.)
        },
        transform: ShapeTransform {
            scale: scale*fox_scale,
            rotate: 0.,
            translate: fox_pos
        },
        color: orange,
        no_smin: true,
        ..Default::default()
    };
    tri5.build();

    let mut circ6 = Shape {
        stype: ShapeType::Circle,
        r: 1.,
        local_transform: ShapeTransform { 
            scale: 1., 
            rotate: 0., 
            translate: Point::new(1., -3.)
        },
        transform: ShapeTransform {
            scale: scale*fox_scale,
            rotate: 0.,
            translate: fox_pos
        },
        color: yellow,
        no_smin: true,
        ..Default::default()
    };
    circ6.build();

    let mut sq7 = Shape {
        stype: ShapeType::Rectangle,
        w: 1.,
        h: 1.,
        local_transform: ShapeTransform { 
            scale: 2_f32.powf(0.5),
            rotate: PI*0.25,
            translate: Point::new(1., -5.)
        },
        transform: ShapeTransform {
            scale: scale*fox_scale,
            rotate: 0.,
            translate: fox_pos
        },
        color: orange,
        no_smin: true,
        ..Default::default()
    };
    sq7.build();

    let mut tri8 = Shape {
        stype: ShapeType::Triangle,
        A: Point::new(0., 0.),
        B: Point::new(1., 0.),
        C: Point::new(1., -1.),
        local_transform: ShapeTransform { 
            scale: 1./(2_f32.powf(0.5)),
            rotate: -PI*0.25,
            translate: Point::new(0., -5.)
        },
        transform: ShapeTransform {
            scale: scale*fox_scale,
            rotate: 0.,
            translate: fox_pos
        },
        color: orange,
        no_smin: true,
        ..Default::default()
    };
    tri8.build();

    let mut tri9 = Shape {
        stype: ShapeType::Triangle,
        A: Point::new(0., 0.),
        B: Point::new(-1., 0.),
        C: Point::new(-1., -1.),
        local_transform: ShapeTransform { 
            scale: 1./(2_f32.powf(0.5)),
            rotate: PI*0.25,
            translate: Point::new(2., -5.)
        },
        transform: ShapeTransform {
            scale: scale*fox_scale,
            rotate: 0.,
            translate: fox_pos
        },
        color: orange,
        no_smin: true,
        ..Default::default()
    };
    tri9.build();

    let mut seg10 = Shape {
        stype: ShapeType::LineSegment,
        A: Point::new(0., 0.),
        B: Point::new(0., -0.3),
        local_transform: ShapeTransform { 
            scale: 1.,
            rotate: 0.,
            translate: Point::new(0., -6.)
        },
        transform: ShapeTransform {
            scale: scale*fox_scale,
            rotate: 0.,
            translate: fox_pos
        },
        no_smin: true,
        ..Default::default()
    };
    seg10.build();

    let mut seg11 = Shape {
        stype: ShapeType::LineSegment,
        A: Point::new(0., 0.),
        B: Point::new(0., -0.3),
        local_transform: ShapeTransform { 
            scale: 1.,
            rotate: 0.,
            translate: Point::new(2., -6.)
        },
        transform: ShapeTransform {
            scale: scale*fox_scale,
            rotate: 0.,
            translate: fox_pos
        },
        no_smin: true,
        ..Default::default()
    };
    seg11.build();


    // ******************************** FLAME

    let bird_pos: Point;
    let bird_scale: f32;
    match scene {
        SceneType::All => {
            bird_pos = Point::new(-scale*5., -scale*5.);
            bird_scale = 2.;
        }
        _ => {
            bird_pos = Point::new(-scale*2.5, -scale*4.);
            bird_scale = 8.;
        }
    }

    let mut bird = Shape {
        stype: ShapeType::Bitmap,
        input_bitmap: loadsdf::loadsdf("resources/bird_large.png"),
        local_transform: ShapeTransform {
            scale: 1., 
            rotate: 0., 
            translate: Point::new(0., 0.)
        },
        transform: ShapeTransform {
            scale: scale*bird_scale, 
            rotate: 0., 
            translate: Point::new(-scale*0.8, -scale*0.8) + bird_pos
        },
        tosdf_scale: 0.006,
        field_scale: 1.,
        color: sky,
        ..Default::default()
    };
    bird.build();
    if scene == SceneType::All || scene == SceneType::Bird {
        bird.bitmap_to_sdf(1., 0.);
    }

    let flame_pos: Point;
    let flame_scale: f32;
    let flame_field: f32;
    match scene {
        SceneType::All => {
            flame_pos = Point::new(scale*4., -scale*4.);
            flame_scale = 3.;
            flame_field = 1.;
        }
        _ => {
            flame_pos = Point::new(-scale*0., -scale*1.);
            flame_scale = 16.;
            flame_field = 1.; // smaller value => smaller distance
        }
    }

    let mut flame = Shape {
        stype: ShapeType::Bitmap,
        bitmap: loadsdf::loadsdf("resources/flame.png"),
        local_transform: ShapeTransform {
            scale: 1., 
            rotate: 0., 
            translate: Point::new(-0.5, -0.5)
        },
        transform: ShapeTransform {
            scale: scale*flame_scale,
            rotate: 0., 
            translate: flame_pos
        },
        color: red,
        gradient_color: yellow,
        gradient_p1: Point::new(0., 0.),
        gradient_p2: Point::new(0.5, 1.),
        field_scale: flame_field,
        layer: 2,
        ..Default::default()
    };
    flame.build();

    let mut flame2 = Shape {
        stype: ShapeType::Bitmap,
        input_bitmap: loadsdf::loadsdf("resources/flame_large.png"),
        local_transform: ShapeTransform {
            scale: 1., 
            rotate: 0., 
            translate: Point::new(-0.5, -0.5)
        },
        transform: ShapeTransform {
            scale: scale*flame_scale, 
            rotate: 0., 
            translate: flame_pos
        },
        color: fur,
        gradient_color: azul,
        gradient_p1: Point::new(0.5, 0.4),
        gradient_p2: Point::new(0.4, 0.5),
        layer: 2,

        tosdf_scale: 0.005,
        field_scale: flame_field,
        ..Default::default()
    };
    flame2.build();
    if scene == SceneType::Flame {
        flame2.bitmap_to_sdf(0.1, 0.5);
    }

    let mut flame3 = Shape {
        stype: ShapeType::Bitmap,
        input_bitmap: loadsdf::loadsdf("resources/flame_large.png"),
        local_transform: ShapeTransform {
            scale: 1., 
            rotate: 0., 
            translate: Point::new(-0.5, -0.5)
        },
        transform: ShapeTransform {
            scale: scale*flame_scale,
            rotate: 0., 
            translate: flame_pos + Point::new(-scale*2.5, scale*2.)
        },
        color: fur,
        gradient_color: azul,
        gradient_p1: Point::new(0.5, 0.4),
        gradient_p2: Point::new(0.4, 0.5),
        layer: 2,

        tosdf_scale: 0.005,
        field_scale: flame_field,
        ..Default::default()
    };
    flame3.build();
    if scene == SceneType::Flame {
        flame3.bitmap_to_sdf(0.1, 0.5);
    }

    let mut flame4 = Shape {
        stype: ShapeType::Bitmap,
        input_bitmap: loadsdf::loadsdf("resources/flame_large.png"),
        local_transform: ShapeTransform {
            scale: 1., 
            rotate: 0., 
            translate: Point::new(-0.5, -0.5)
        },
        transform: ShapeTransform {
            scale: scale*flame_scale,
            rotate: 0., 
            translate: flame_pos + Point::new(scale*2.5, scale*2.)
        },
        color: fur,
        gradient_color: azul,
        gradient_p1: Point::new(0.5, 0.4),
        gradient_p2: Point::new(0.4, 0.5),
        layer: 2,

        tosdf_scale: 0.005,
        field_scale: flame_field,
        ..Default::default()
    };
    flame4.build();
    if scene == SceneType::Flame {
        flame4.bitmap_to_sdf(0.1, 0.5);
    }


    let cor2_pos = Point::new(scale*4.5, scale*5.);
    let a_scale = 0.7;
    let a_r = 0.6;

    let mut a_circ1 = Shape {
        stype: ShapeType::Circle,
        r: 1.,
        local_transform: ShapeTransform { 
            scale: 1.,
            rotate: 0.,
            translate: Point::new(a_scale, 0.)
        },
        transform: ShapeTransform {
            scale: scale*a_r,
            rotate: 0.,
            translate: cor2_pos
        },
        // no_smin: true,
        color: orange,
        layer: 3,
        ..Default::default()
    };
    a_circ1.build();

    let mut a_circ2 = Shape {
        stype: ShapeType::Circle,
        r: 1.,
        local_transform: ShapeTransform { 
            scale: 1.,
            rotate: 0.,
            translate: Point::new(-a_scale, 0.)
        },
        transform: ShapeTransform {
            scale: scale*a_r,
            rotate: 0.,
            translate: cor2_pos
        },
        // no_smin: true,
        color: yellow,
        layer: 3,
        ..Default::default()
    };
    a_circ2.build();

    let mut a_circ3 = Shape {
        stype: ShapeType::Circle,
        r: 1.,
        local_transform: ShapeTransform { 
            scale: 1.,
            rotate: 0.,
            translate: Point::new(0., -a_scale*(3_f32).powf(0.5))
        },
        transform: ShapeTransform {
            scale: scale*a_r,
            rotate: 0.,
            translate: cor2_pos
        },
        // no_smin: true,
        color: leaf,
        layer: 3,
        ..Default::default()
    };
    a_circ3.build();


    let mut shapes: Vec<Shape> = Vec::new();

    {
        use SceneType::*;
        if scene == All || scene == Letters {
            shapes.push(letterm);
            shapes.push(letterv);
        }

        // Fox
        if scene == All || scene == Fox {
            shapes.push(tri1);
            shapes.push(rect2);
            shapes.push(tri3);
            shapes.push(tri4);
            shapes.push(tri5);
            shapes.push(circ6);
            shapes.push(sq7);
            shapes.push(tri8);
            shapes.push(tri9);
            shapes.push(seg10);
            shapes.push(seg11);
        }

        if scene == Flame {
            shapes.push(flame2);
            shapes.push(flame3);
            shapes.push(flame4);
        }

        if scene == All || scene == Bird {
            shapes.push(bird);
        }

        if scene == All {
            shapes.push(flame);
            shapes.push(a_circ1);
            shapes.push(a_circ2);
            shapes.push(a_circ3);
        }

        if scene == Shapes {
            shapes.push(s_circ1);
            shapes.push(s_circ2);
            shapes.push(s_circ3);
            shapes.push(s_circ4);
            shapes.push(s_tri);
            shapes.push(s_tri2);
            shapes.push(s_tri3);
            shapes.push(s_tri4);
        }

        if scene == Art {

            for _i in 0..20 {
                use ShapeType::*;
                let stn = (rand::random::<f32>() * 6.) as u32;
                let st: ShapeType;
                match stn {
                    0 => st = Circle,
                    1 => st = Rectangle,
                    2 => st = Triangle,
                    _ => st = LineSegment
                }
                let rad = rand::random::<f32>() * 4.;
                let w = rand::random::<f32>() * 10.;
                let h = rand::random::<f32>() * 10./w;
                let a = Point::new(rand::random::<f32>()-0.5, rand::random::<f32>()-0.5)*5.;
                let b = Point::new(rand::random::<f32>()-0.5, rand::random::<f32>()-0.5)*5.;
                let c = Point::new(rand::random::<f32>()-0.5, rand::random::<f32>()-0.5)*5.;
                let mut sc = (1. + (rand::random::<f32>()-0.5))*0.6;
                if st == Circle {
                    sc /= 5.;
                }
                if st == LineSegment {
                    sc *= 5.;
                }
                let rot = rand::random::<f32>()*PI*2.;
                let tr = Point::new((rand::random::<f32>()-0.5) * 6., (rand::random::<f32>()-0.5) * 6.)*scale;
                
                let coln = (rand::random::<f32>() * 10.) as u32;
                let col: ShapeColor;
                col = match coln {
                    0 => yellow,
                    1 => orange,
                    2 => pink,
                    3 => red,
                    4 => sky,
                    5 => grey,
                    6 => leaf,
                    7 => grass,
                    _ => fur
                };

                //println!("Shape type {st:?} w {w} h {h} sc {sc} rot {rot} tr {tr:?}");

                let mut some_shape = Shape {
                    stype: st,
                    r: rad,
                    w: w,
                    h: h,
                    A: a,
                    B: b,
                    C: c,
                    local_transform: ShapeTransform { 
                        scale: 1.,
                        rotate: 0.,
                        translate: Point::new(0., 0.)
                    },
                    transform: ShapeTransform {
                        scale: scale*sc,
                        rotate: rot,
                        translate: tr
                    },
                    color: col,
                    ..Default::default()
                };
                some_shape.build();
                shapes.push(some_shape);
            }
        }
    }

    println!("Shapes N={}", shapes.len());



    let mut dist: Vec<f32> = vec![0.; shapes.len()];

    println!("Initialization took {:?}", start_time.elapsed());
    start_time = Instant::now();

    let imgx = 1600;
    let imgy = imgx;

    // Create a new ImgBuf with width: imgx and height: imgy
    let mut imgbuf = image::ImageBuffer::new(imgx, imgy);

    // Iterate over the coordinates and pixels of the image
    for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
        let xf = x as f32 / imgx as f32 - 0.5;
        let yf = y as f32 / imgy as f32 - 0.5;
        //let r = (xf / imgx as f32 * 128.0) as u8;
        //let b = (yf / imgy as f32 * 128.0) as u8;

        let p: Point = Point::new(xf, yf);
        //let r = distance(Point::new(xf, yf), Point::new(0.05, 0.5)) * 2.0 * 255.0;
        //let r = tri.distance(Point::new(xf, yf)) * 3.0 * 255.0;
        let mut r: f32 = 1000.;
        let mut min_idx: usize = 0;
        for i in 0..shapes.len() {
            let ri = shapes[i].distance(p);
            if ri < r {
                min_idx = i;
            }
            if shapes[i].no_smin {
                r = r.min(ri);
            } else {
                r = smin(r, ri, scale/3.);
            }
            dist[i] = ri;
        }

        let mut aliasing_scale = main_aliasing_scale;
        let mut border_scale = main_border_scale;
        if shapes[min_idx].layer == 1 || scene == SceneType::Bird {
            aliasing_scale = letter_aliasing_scale;
            border_scale = letter_border_scale;
        } else if shapes[min_idx].layer == 2 {
            aliasing_scale = 0.003*4.;
            border_scale = 0.004*4.;
        }


        let back_gradient: ShapeColor;
        if scene == SceneType::Art || scene == SceneType::Bird {
            back_gradient = ShapeColor{r: 1., g: 1., b: 1., layer: 0};
        } else if scene == SceneType::Flame {
            back_gradient = ShapeColor{r: 0., g: 0., b: 0., layer: 0};
        } else {
            back_gradient = ShapeColor{r:240./255.-xf.abs(), g:202./255.-xf, b:yf.abs()*3., layer:0};
        }
        let mut out_color: ShapeColor;
        if r < 0. || r < aliasing_scale {
            let mut blend: ShapeColor = ShapeColor::new(0, 0, 0);

            let mut eps = scale / 3.;
            if shapes[min_idx].layer == 3 {
                eps = scale / 1.5;
            }
            if scene == SceneType::Shapes {
                eps = scale / 1.;
            }
            if scene == SceneType::Flame {
                eps = scale / 0.4;
            }
            let mut bleda_sum: f32 = 0.;
            for i in 0..shapes.len() {
                let ri = dist[i];
                if ri - r < eps {
                    bleda_sum += r + eps - ri;
                }
            }

            for i in 0..shapes.len() {
                let ri = dist[i];
                if ri - r < eps {
                    let mut cur_color = shapes[i].color; // TODO: ugly, please remove this hard-code
                    if shapes[i].layer == 2 {
                        cur_color = shapes[i].gradient(p);
                    }
                    blend = blend + cur_color * ((r + eps - ri) / bleda_sum);
                }
            }

            out_color = blend;

            if 0. < r && r < aliasing_scale {
                let k_inside = (aliasing_scale - r) / aliasing_scale;
                out_color = out_color * k_inside + grey * (1.-k_inside);
            }
        } else if r < border_scale {
            out_color = grey;
        } else if r < border_scale + aliasing_scale {
            let k_outside = (r - border_scale) / aliasing_scale;
            out_color = grey * (1.-k_outside) + back_gradient * (k_outside);
        } else {
            out_color = back_gradient;
        }


        out_color = out_color * 255.;
        *pixel = image::Rgb([
            out_color.r as u8,
            out_color.g as u8,
            out_color.b as u8]);
    }

    println!("Rendering took {:?}", start_time.elapsed());
    start_time = Instant::now();

    // Save the image as “fractal.png”, the format is deduced from the path
    imgbuf.save(save_path).unwrap();

    println!("Saving took {:?}", start_time.elapsed());
}


fn main() {
    main_letters(SceneType::Shapes, "fractal.png");
    main_letters(SceneType::All, "fractal0.png");
    main_letters(SceneType::Bird, "fractal1.png");
    main_letters(SceneType::Fox, "fractal2.png");
    main_letters(SceneType::Letters, "fractal3.png");
    main_letters(SceneType::Flame, "fractal4.png");
    main_letters(SceneType::Art, "fractal5.png");
    
    return;
}