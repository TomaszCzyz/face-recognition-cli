use blake3::Hash;
use sqlx::types::chrono::{DateTime, Utc};
use dlib_wrappers::face_encoding::FaceEncoding;

pub trait PersonRegistry {
    async fn find_file(&self, hash: &Hash) -> Option<(i32, String, DateTime<Utc>)>;
    fn get(&self, encoding: &FaceEncoding) -> Option<String>;
    fn add(&mut self, encoding: FaceEncoding, name: String);
}
