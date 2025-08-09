use once_cell::sync::Lazy;
use opentelemetry::global;
use opentelemetry::metrics::Histogram;
use opentelemetry_sdk::metrics::SdkMeterProvider;

pub(crate) static HISTOGRAM_F_D: Lazy<Histogram<u64>> = Lazy::new(|| {
    global::meter("face_recognizer")
        .u64_histogram("face_recognition.duration_ms")
        .with_description("Face recognition execution time in ms")
        .with_boundaries(vec![
            0.0, 250.0, 1000.0, 5000.0, 6000.0, 10000.0, 20000.0, 30000.0, 40000.0, 50000.0,
            60000.0,
        ])
        .build()
});

pub(crate) static HISTOGRAM_L_P: Lazy<Histogram<u64>> = Lazy::new(|| {
    global::meter("face_recognizer")
        .u64_histogram("landmarks_prediction.duration_ms")
        .with_description("Landmarks prediction execution time in ms")
        .with_boundaries(vec![0.0, 5.0, 12.5, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0])
        .build()
});

pub(crate) static HISTOGRAM_F_E: Lazy<Histogram<u64>> = Lazy::new(|| {
    global::meter("face_recognizer")
        .u64_histogram("face_encoding.duration_ms")
        .with_description("Face encoding generation time in ms")
        .with_boundaries(vec![0.0, 5.0, 12.5, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0])
        .build()
});

pub fn init_metrics() -> SdkMeterProvider {
    let exporter = opentelemetry_stdout::MetricExporter::default();
    let provider = SdkMeterProvider::builder()
        .with_periodic_exporter(exporter)
        .build();

    global::set_meter_provider(provider.clone());
    provider
}
