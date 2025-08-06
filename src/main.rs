#![allow(dead_code)]

use dlib_wrappers::face_detection::{FaceDetector, FaceDetectorCnn, FaceDetectorModel};
use dlib_wrappers::landmark_prediction::LandmarkPredictor;
use dlib_wrappers::{ImageMatrix, Point, Rectangle};
use image::{Rgb, RgbImage, open};
use std::time::Instant;

fn main() {
    let cmd = clap::Command::new("face-recognizer")
        .subcommand_required(true)
        .subcommand(
            clap::command!("recognize").args(&[
                clap::arg!(<input> "a path to a file to analyse")
                    .value_parser(clap::value_parser!(std::path::PathBuf)),
                clap::arg!(--"names-path" <PATH>)
                    .value_parser(clap::value_parser!(std::path::PathBuf)),
            ]),
        );

    let matches = cmd.get_matches();
    match matches.subcommand() {
        Some(("recognize", matches)) => {
            let input = matches.get_one::<std::path::PathBuf>("input").unwrap();

            let input_filename = input.file_stem().unwrap().to_str().unwrap();
            let input_ext = input.extension().unwrap().to_str().unwrap();
            let output = input
                .parent()
                .unwrap()
                .join(format!("{}_new.{}", input_filename, input_ext));

            let mut image = open(input).unwrap().to_rgb8();

            let models = DefaultModels::default();

            let green = Rgb([0, 255, 0]);

            detect_and_mark(
                &mut image,
                &models.cnn_detector,
                &models.landmarks,
                Some(green),
            );

            image.save(&output).unwrap();
        }
        _ => unreachable!("clap should ensure we don't get here"),
    };
}

fn detect_and_mark(
    image: &mut RgbImage,
    face_detector: &dyn FaceDetectorModel,
    landmarks: &LandmarkPredictor,
    color: Option<Rgb<u8>>,
) {
    println!("starting...");

    let color = color.unwrap_or(Rgb([255, 0, 0]));
    let matrix = ImageMatrix::from_image(&image);

    let face_locations_start = Instant::now();
    let face_locations = face_detector.face_locations(&matrix);
    let _ = face_locations_start.elapsed();

    println!("finding faces took: {:?}", face_locations_start.elapsed());

    for r in face_locations.iter() {
        draw_rectangle(image, &r, color);

        let landmarks_start = Instant::now();
        let landmarks = landmarks.face_landmarks(&matrix, &r);
        let _ = landmarks_start.elapsed();
        println!("finding landmarks took: {:?}", landmarks_start.elapsed());

        for point in landmarks.iter() {
            draw_point(image, &point, color);
        }
    }

    println!("finished");
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
}

struct DefaultModels {
    detector: FaceDetector,
    cnn_detector: FaceDetectorCnn,
    landmarks: LandmarkPredictor,
}

impl Default for DefaultModels {
    fn default() -> Self {
        Self {
            detector: FaceDetector::default(),
            cnn_detector: FaceDetectorCnn::new(
                "crates/dlib_wrappers/files/mmod_human_face_detector.dat",
            )
            .expect("Failed to load CNN face detector"),
            landmarks: LandmarkPredictor::new(
                "crates/dlib_wrappers/files/shape_predictor_68_face_landmarks.dat",
            )
            .expect("Failed to load landmark predictor"),
        }
    }
}
