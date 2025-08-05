use dlib_wrappers::face_detection::{FaceDetector, FaceDetectorCnn};
use dlib_wrappers::landmark_prediction::LandmarkPredictor;
use dlib_wrappers::{ImageMatrix, Point, Rectangle};
use image::{Rgb, RgbImage, open};

fn main() {
    let input = "./assets/pink_party.jpeg";
    let output = "./assets/pink_party_new.jpeg";

    let mut image = open(input).unwrap().to_rgb8();
    let matrix = ImageMatrix::from_image(&image);

    let detector = FaceDetector::default();

    let cnn_detector = FaceDetectorCnn::new("crates/dlib_wrappers/files/mmod_human_face_detector.dat")
        .expect("Failed to load CNN face detector");

    let landmarks = LandmarkPredictor::new("crates/dlib_wrappers/files/shape_predictor_68_face_landmarks.dat")
        .expect("Failed to load landmark predictor");

    let red = Rgb([255, 0, 0]);
    let green = Rgb([0, 255, 0]);

    let face_locations = detector.face_locations(&matrix);

    for r in face_locations.iter() {
        draw_rectangle(&mut image, &r, red);

        let landmarks = landmarks.face_landmarks(&matrix, &r);

        for point in landmarks.iter() {
            draw_point(&mut image, &point, red);
        }
    }

    let face_locations = cnn_detector.face_locations(&matrix);

    for r in face_locations.iter() {
        draw_rectangle(&mut image, &r, green);

        let landmarks = landmarks.face_landmarks(&matrix, &r);

        for point in landmarks.iter() {
            let p = Point {
                x: (image.width() as i64 - 1).min(point.x),
                y: (image.height() as i64 - 1).min(point.y),
            };
            draw_point(&mut image, &p, green);
        }
    }

    image.save(&output).unwrap();
}

fn draw_rectangle(image: &mut RgbImage, rect: &Rectangle, colour: Rgb<u8>) {
    for x in rect.left..rect.right {
        image.put_pixel(x as u32, rect.top as u32, colour);
        image.put_pixel(x as u32, rect.bottom as u32, colour);
    }

    for y in rect.top..rect.bottom {
        image.put_pixel(rect.left as u32, y as u32, colour);
        image.put_pixel(rect.right as u32, y as u32, colour);
    }
}

fn draw_point(image: &mut RgbImage, point: &Point, colour: Rgb<u8>) {
    image.put_pixel(point.x as u32, point.y as u32, colour);
    // image.put_pixel(point.x as u32 + 1, point.y as u32, colour);
    // image.put_pixel(point.x as u32 + 1, point.y as u32 + 1, colour);
    // image.put_pixel(point.x as u32, point.y as u32 + 1, colour);
}
