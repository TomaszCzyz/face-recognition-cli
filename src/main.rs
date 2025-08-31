#![allow(dead_code)]

use crate::face_recognizer::{FaceRecognizer, FaceRecognizerOptions};
use crate::person_registry::person_registry_sqlite::PersonRegistrySqlite;
use clap::{Arg, ArgAction};
use directories::ProjectDirs;
use dlib_wrappers::face_detection::FaceDetectorCnn;
use dlib_wrappers::face_encoding::FaceEncodingNetwork;
use dlib_wrappers::landmark_prediction::LandmarkPredictor;
use futures::stream::{self, StreamExt};
use indicatif::ProgressState;
use once_cell::sync::Lazy;
use std::fs::DirEntry;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::{fs, io};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use tracing::level_filters::LevelFilter;
use tracing::{error, info};
use tracing_indicatif::IndicatifLayer;
use tracing_indicatif::style::ProgressStyle;
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let worker_threads = (cores / 2).max(1);

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(worker_threads)
        .enable_all()
        .build()?;

    rt.block_on(async {
        main_inner().await?;
        Ok::<_, Box<dyn std::error::Error>>(())
    })?;

    Ok(())
}

async fn main_inner() -> Result<(), Box<dyn std::error::Error>> {
    // force early init of the project dirs to handle panic
    let _ = PROJECT_DIRS.data_dir();

    let indicatif_layer = IndicatifLayer::new().with_progress_style(
        ProgressStyle::with_template(
            "{color_start}{span_child_prefix}{span_fields} -- {span_name} {wide_msg} {elapsed_subsec}{color_end}",
        )
            .unwrap()
            .with_key(
                "elapsed_subsec",
                elapsed_subsec,
            )
            .with_key(
                "color_start",
                |state: &ProgressState, writer: &mut dyn std::fmt::Write| {
                    let elapsed = state.elapsed();

                    if elapsed > Duration::from_secs(30) {
                        // Red
                        let _ = write!(writer, "\x1b[{}m", 1 + 30);
                    } else if elapsed > Duration::from_secs(15) {
                        // Yellow
                        let _ = write!(writer, "\x1b[{}m", 3 + 30);
                    }
                },
            )
            .with_key(
                "color_end",
                |state: &ProgressState, writer: &mut dyn std::fmt::Write| {
                    if state.elapsed() > Duration::from_secs(4) {
                        let _ =write!(writer, "\x1b[0m");
                    }
                },
            ),
    ).with_span_child_prefix_symbol("↳ ").with_span_child_prefix_indent(" ");

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(indicatif_layer.get_stderr_writer())
                .event_format(
                    format()
                        .with_level(false)
                        .with_target(false)
                        .without_time()
                        .with_source_location(false),
                ),
        )
        .with(indicatif_layer)
        .with(LevelFilter::INFO)
        .init();

    // let provider = init_metrics();

    let persons_registry = PersonRegistrySqlite::initialize().await;

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
                // Limit concurrency to something reasonable. You can tune this.
                let max_concurrency = std::thread::available_parallelism()
                    .map(|n| n.get() / 2)
                    .unwrap_or(1);

                let semaphore = Arc::new(Semaphore::new(max_concurrency));
                let mut join_set = JoinSet::new();

                for entry in WalkDir::new(input)
                    .into_iter()
                    .filter_map(|e| e.ok())
                    .filter(|e| e.path().is_file())
                {
                    let sqlite = persons_registry.clone();
                    let permit = semaphore.clone().acquire_owned().await.unwrap();
                    let path = entry.path().to_path_buf();

                    let models = Arc::new(DefaultModels::default());

                    join_set.spawn(async move {
                        // Hold the permit for the duration of the task
                        let _permit = permit;
                        let models = models.clone();

                        let mut recognizer = FaceRecognizer::new(models, sqlite);

                        recognizer.process_file(&path, options).await;
                    });
                }

                while let Some(res) = join_set.join_next().await {
                    if let Err(err) = res {
                        error!("task failed: {}", err);
                    }
                }
            } else if input.is_file() {
                let models = DefaultModels::default();
                let mut recognizer = FaceRecognizer::new(models.into(), persons_registry.clone());

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

// async fn taker<F>(f: fn(One) -> F)
// where
//     F: Future<Output = Two>,
// {
//     let two = f(One).await;
// }

// one possible implementation of walking a directory only visiting files
async fn visit_dirs(dir: &Path, cb: &mut dyn FnMut(&DirEntry)) -> io::Result<()> {
    if dir.is_dir() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                Box::pin(visit_dirs(&path, cb)).await?;
            } else {
                cb(&entry);
            }
        }
    }
    Ok(())
}
