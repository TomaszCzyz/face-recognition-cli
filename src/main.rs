#![allow(dead_code)]

use crate::face_recognizer::{DetectResult, FaceRecognizer, FaceRecognizerOptions};
use crate::person_registry::person_registry_sqlite::PersonRegistrySqlite;
use clap::{Arg, ArgAction};
use directories::ProjectDirs;
use dlib_wrappers::face_detection::FaceDetectorCnn;
use dlib_wrappers::face_encoding::FaceEncodingNetwork;
use dlib_wrappers::landmark_prediction::LandmarkPredictor;
use indicatif::ProgressState;
use once_cell::sync::Lazy;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tracing::info;
use tracing::level_filters::LevelFilter;
use tracing_subscriber::fmt::format;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use walkdir::WalkDir;

mod face_recognizer;
mod image_helpers;
mod otel;
mod person_registry;

static PROJECT_DIRS: Lazy<ProjectDirs> = Lazy::new(|| {
    ProjectDirs::from("com", "example", "face-recognizer")
        .expect("failed to determine project directories")
});

fn elapsed_subsec(state: &ProgressState, writer: &mut dyn std::fmt::Write) {
    let seconds = state.elapsed().as_secs();
    let sub_seconds = (state.elapsed().as_millis() % 1000) / 100;
    let _ = writer.write_str(&format!("{}.{}s", seconds, sub_seconds));
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // force early init of the project dirs to handle panic
    let _ = PROJECT_DIRS.data_dir();

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer().event_format(
                format()
                    .with_level(false)
                    .with_target(false)
                    .without_time()
                    .with_source_location(false),
            ),
        )
        .with(LevelFilter::INFO)
        .init();

    let persons_registry = PersonRegistrySqlite::initialize().await;
    let models = Arc::new(DefaultModels::default());
    let recognizer = FaceRecognizer::new(models.clone(), persons_registry.clone());

    let cmd = clap::Command::new("face-recognizer")
        .subcommand_required(true)
        .subcommand(
            clap::command!("recognize").args(&[
                clap::arg!(<input> "a path to a file to analyse")
                    .value_parser(clap::value_parser!(PathBuf)),
                clap::arg!(--"names-path" <PATH>).value_parser(clap::value_parser!(PathBuf)),
                Arg::new("skip-processed-check")
                    .long("skip-processed-check")
                    .action(ArgAction::SetTrue),
            ]),
        )
        .subcommand(clap::command!("locate").args(&[
            clap::arg!(<ID> "a path to a file to analyse").value_parser(clap::value_parser!(i64)),
        ]));

    match cmd.get_matches().subcommand() {
        Some(("recognize", matches)) => {
            let recognize_start = Instant::now();

            let input = matches.get_one::<PathBuf>("input").unwrap();
            let skip_processed_check = matches.get_flag("skip-processed-check");

            let options = FaceRecognizerOptions {
                skip_processed_check,
            };

            if input.is_dir() {
                for entry in WalkDir::new(input)
                    .into_iter()
                    .filter_map(|e| e.ok())
                    .filter(|e| e.path().is_file())
                {
                    let path = entry.path().to_path_buf();
                    match recognizer.process_file(&path, options).await {
                        DetectResult::Skipped => {}
                        DetectResult::NoFaces => info!("no faces found in {}", path.display()),
                        DetectResult::FacesDetected(face_ids) => {
                            info!("found {} faces in {}", face_ids.len(), path.display());

                            for face_id in face_ids {
                                let similar_face_ids = persons_registry.locate_similar(face_id).await;

                                if !similar_face_ids.is_empty() {
                                    info!("I do not recognize the person.. Could you tell who that is?");
                                    // persons_registry.
                                } else {
                                    info!("no similar faces found for face id {}", face_id);
                                }
                            }
                        }
                    }
                }
            } else if input.is_file() {
                recognizer.process_file(&input, options).await;
            }

            info!(
                "recognizing faces finished in: {:?}",
                recognize_start.elapsed()
            );
        }
        Some(("locate", matches)) => {
            let encoding_id = *matches.get_one::<i64>("ID").unwrap();
            persons_registry.locate_similar(encoding_id).await;
        }
        _ => unreachable!("clap should ensure we don't get here"),
    };

    // provider.shutdown()?;

    println!("done");
    Ok(())
}

fn get_output_path(input: &Path) -> PathBuf {
    let input_filename = input.file_stem().unwrap().to_str().unwrap();
    let input_ext = input.extension().unwrap().to_str().unwrap();
    let output = input
        .parent()
        .unwrap()
        .join(format!("{}_new.{}", input_filename, input_ext));

    output
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
