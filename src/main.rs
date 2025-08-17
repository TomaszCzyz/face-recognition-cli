#![allow(dead_code)]
use crate::otp::{HISTOGRAM_F_D, HISTOGRAM_F_E, HISTOGRAM_L_P, init_metrics};
use crate::person_registry::person_registry_file::PersonRegistryFile;
use directories::ProjectDirs;
use dlib_wrappers::face_detection::{FaceDetectorCnn, FaceDetectorModel};
use dlib_wrappers::face_encoding::FaceEncodingNetwork;
use dlib_wrappers::landmark_prediction::LandmarkPredictor;
use dlib_wrappers::{ImageMatrix, Point, Rectangle};
use image::{Rgb, RgbImage, open};
use opentelemetry::KeyValue;
use opentelemetry::metrics::MeterProvider;
use std::path::PathBuf;
use std::time::Instant;
use uuid::Uuid;

mod face_recognizer;
mod otp;
mod person_registry;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider = init_metrics();

    let cmd = clap::Command::new("face-recognizer")
        .subcommand_required(true)
        .subcommand(
            clap::command!("recognize").args(&[
                clap::arg!(<input> "a path to a file to analyse")
                    .value_parser(clap::value_parser!(PathBuf)),
                clap::arg!(--"names-path" <PATH>).value_parser(clap::value_parser!(PathBuf)),
            ]),
        );

    let models = DefaultModels::default();

    let file_name = if let Some(proj_dirs) = ProjectDirs::from("com", "Example", "face-recognizer")
    {
        proj_dirs.data_dir().join("persons_registry.csv")
    } else {
        panic!("Failed to get project dirs");
    };

    let mut persons_registry = PersonRegistryFile::new(&file_name);

    let matches = cmd.get_matches();
    match matches.subcommand() {
        Some(("recognize", matches)) => {
            let input = matches.get_one::<PathBuf>("input").unwrap();

            if input.is_dir() {
                for dir_entry in std::fs::read_dir(input)
                    .unwrap()
                    .filter(|x| x.as_ref().is_ok_and(|x| x.path().is_file()))
                {
                    let path = &dir_entry.unwrap().path();

                    println!("processing: {:?}", path);
                    let mut image = open(path).unwrap().to_rgb8();

                    detect_and_mark(&mut image, &models, &mut persons_registry);

                    let output = get_output_path(path);
                    image.save(&output).unwrap();
                }
            } else if input.is_file() {
                println!("processing: {:?}", input);
                let mut image = open(input).unwrap().to_rgb8();

                detect_and_mark(&mut image, &models, &mut persons_registry);

                let output = get_output_path(input);
                image.save(&output).unwrap();
            }
        }
        _ => unreachable!("clap should ensure we don't get here"),
    };

    persons_registry.save();
    provider.shutdown()?;
    // let t = provider.shutdown();

    Ok(())
}

fn get_output_path(input: &PathBuf) -> PathBuf {
    let input_filename = input.file_stem().unwrap().to_str().unwrap();
    let input_ext = input.extension().unwrap().to_str().unwrap();
    let output = input
        .parent()
        .unwrap()
        .join(format!("{}_new.{}", input_filename, input_ext));
    output
}

fn detect_and_mark(
    image: &mut RgbImage,
    models: &DefaultModels,
    persons_registry: &mut PersonRegistryFile,
) {
    let start = Instant::now();
    println!("starting...");

    let color = Rgb([255, 0, 0]);
    let matrix = ImageMatrix::from_image(&image);

    let face_locations_start = Instant::now();
    let face_locations = models.face_detector.face_locations(&matrix);
    println!(
        "found {:?} faces in {:?}",
        face_locations.len(),
        face_locations_start.elapsed()
    );

    HISTOGRAM_F_D.record(
        face_locations_start.elapsed().as_millis() as u64,
        &[KeyValue::new("file", "file_name")],
    );

    for r in face_locations.iter() {
        draw_rectangle(image, &r, color);

        let landmarks_start = Instant::now();
        let landmarks = models.landmarks_predictor.face_landmarks(&matrix, &r);
        println!("finding landmarks took: {:?}", landmarks_start.elapsed());
        HISTOGRAM_L_P.record(
            landmarks_start.elapsed().as_millis() as u64,
            &[KeyValue::new("file", "file_name")],
        );

        let face_encoding_start = Instant::now();
        let encodings = models
            .face_encoding
            .get_face_encodings(&matrix, &[landmarks.clone()], 0);
        println!(
            "calculating encodings took: {:?}",
            face_encoding_start.elapsed()
        );
        let id = Uuid::new_v4();
        HISTOGRAM_F_E.record(
            face_encoding_start.elapsed().as_millis() as u64,
            &[
                KeyValue::new("file", "file_name"),
                KeyValue::new("id", id.to_string()),
            ],
        );

        for encoding in encodings.iter() {
            if let Some(name) = persons_registry.get(encoding) {
                println!("found person in registry: {}", name);
            } else {
                let id = Uuid::new_v4();
                persons_registry.add(encoding.clone(), id.to_string());
            }
        }

        for point in landmarks.iter() {
            draw_point(image, &point, color);
        }
    }

    println!("finished {:?}", start.elapsed());
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
    face_detector: FaceDetectorCnn,
    landmarks_predictor: LandmarkPredictor,
    face_encoding: FaceEncodingNetwork,
}

impl Default for DefaultModels {
    fn default() -> Self {
        Self {
            // detector: FaceDetector::default(),
            face_detector: FaceDetectorCnn::new(
                "crates/dlib_wrappers/files/mmod_human_face_detector.dat",
            )
            .expect("Failed to load CNN face detector"),
            landmarks_predictor: LandmarkPredictor::new(
                "crates/dlib_wrappers/files/shape_predictor_68_face_landmarks.dat",
            )
            .expect("Failed to load landmark predictor"),
            face_encoding: FaceEncodingNetwork::new(
                "crates/dlib_wrappers/files/dlib_face_recognition_resnet_model_v1.dat",
            )
            .expect("Failed to load face encoding network"),
        }
    }
}
