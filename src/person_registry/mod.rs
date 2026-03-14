use blake3::Hash;
use dlib_wrappers::face_encoding::FaceEncoding;
use sqlx::types::chrono::{DateTime, Utc};

pub(crate) mod person_registry_sqlite;

pub trait PersonRegistry {
    async fn find_file(&self, hash: &Hash) -> Option<(i32, String, DateTime<Utc>)>;
    fn get(&self, encoding: &FaceEncoding) -> Option<String>;
    fn add(&mut self, encoding: FaceEncoding, name: String);
}
