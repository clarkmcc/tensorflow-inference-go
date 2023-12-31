// Code generated by scale-signature 0.4.3, DO NOT EDIT.
// output: local_bindings_latest_guest

pub mod types;
use crate::types::{Encode, Decode};
use std::io::Cursor;
use polyglot_rs::Encoder;
static HASH: &'static str = "86026d7c55f2c1ca3d712c6ac564b06a76695c40c1eae1b3d46000ad13177961";
static mut READ_BUFFER: Vec<u8> = Vec::new();
static mut WRITE_BUFFER: Vec<u8> = Vec::new();
pub unsafe fn write(ctx: Option<&mut types::Context>) -> (u32, u32) {
    let mut cursor = Cursor::new(Vec::new());
    match ctx {
        Some(ctx) => {
            cursor = match types::Context::encode(Some(ctx), &mut cursor) {
                Ok(_) => cursor,
                Err(err) => return error(err),
            };
        }
        None => {
            cursor = match types::Context::encode(None, &mut cursor) {
                Ok(_) => cursor,
                Err(err) => return error(err),
            };
        }
    }
    let vec = cursor.into_inner();
    WRITE_BUFFER.resize(vec.len() as usize, 0);
    WRITE_BUFFER.copy_from_slice(&vec);
    return (WRITE_BUFFER.as_ptr() as u32, WRITE_BUFFER.len() as u32);
}
pub unsafe fn read() -> Result<Option<types::Context>, Box<dyn std::error::Error>> {
    let mut cursor = Cursor::new(&mut READ_BUFFER);
    types::Context::decode(&mut cursor)
}
pub unsafe fn error(error: Box<dyn std::error::Error>) -> (u32, u32) {
    let mut cursor = Cursor::new(Vec::new());
    return match cursor.encode_error(error) {
        Ok(_) => {
            let vec = cursor.into_inner();
            WRITE_BUFFER.resize(vec.len() as usize, 0);
            WRITE_BUFFER.copy_from_slice(&vec);
            (WRITE_BUFFER.as_ptr() as u32, WRITE_BUFFER.len() as u32)
        }
        Err(_) => (0, 0),
    };
}
pub unsafe fn resize(size: u32) -> *const u8 {
    READ_BUFFER.resize(size as usize, 0);
    return READ_BUFFER.as_ptr();
}
pub unsafe fn hash() -> (u32, u32) {
    let mut cursor = Cursor::new(Vec::new());
    return match cursor.encode_string(&String::from(HASH)) {
        Ok(_) => {
            let vec = cursor.into_inner();
            WRITE_BUFFER.resize(vec.len() as usize, 0);
            WRITE_BUFFER.copy_from_slice(&vec);
            (WRITE_BUFFER.as_ptr() as u32, WRITE_BUFFER.len() as u32)
        }
        Err(_) => (0, 0),
    };
}
pub fn next(
    ctx: Option<types::Context>,
) -> Result<Option<types::Context>, Box<dyn std::error::Error>> {
    unsafe {
        let (ptr, len) = match ctx {
            Some(mut ctx) => write(Some(&mut ctx)),
            None => write(None),
        };
        _next(ptr, len);
        read()
    }
}
#[link(wasm_import_module = "env")]
extern "C" {
    #[link_name = "next"]
    fn _next(ptr: u32, size: u32);
}
