use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Linear, VarBuilder};
use signature::types;

/// The model's weights and biases, exported and stored using the safetensors format
const SAFETENSORS: &[u8] = include_bytes!("../model.safetensors");

/// Our generated [scale](https://scale.sh/) function which is called by our WebAssembly
/// host runtime when it wants to execute our model.
pub fn scale(
    ctx: Option<types::Context>,
) -> Result<Option<types::Context>, Box<dyn std::error::Error>> {
    if let Some(mut ctx) = ctx {
        let model = Model::new(128);
        ctx.digit = model.predict(&ctx.pixels);
        return signature::next(Some(ctx));
    }
    signature::next(ctx)
}

struct Model {
    /// Stores the imported safetensor weights and biases
    vb: VarBuilder<'static>,

    /// The dimension of the hidden layer
    hidden_dim: usize,
}

impl Model {
    /// Creates a new instance of our model and accepts the hidden layer dimension that should be
    /// used for this model. When you train the model, the size of the dense layer should be reflected
    /// and updated here. For example, in the following code, our hidden layer dimension is 128:
    ///
    /// ```python
    /// model = tf.keras.models.Sequential([
    ///      tf.keras.layers.Flatten(input_shape=(28, 28)),
    ///      tf.keras.layers.Dense(128, activation='relu'),
    ///      tf.keras.layers.Dropout(0.2),
    ///      tf.keras.layers.Dense(10)
    ///  ])
    /// ```
    fn new(hidden_dim: usize) -> Self {
        let tensors = candle_core::safetensors::load_buffer(SAFETENSORS, &Device::Cpu).unwrap();
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
        Self { vb, hidden_dim }
    }

    /// Predict accepts a pre-flattened 28x28 grid of pixels and returns the predicted digit.
    fn predict(&self, pixels: &[u32]) -> u32 {
        let pixel_count = 28 * 28; // 784
        let digits_count = 10; // 0-9

        // Convert the input from 0-255 to 0.0-1.0
        let pixels = pixels
            .iter()
            .map(|p| *p as f32 / 255.0)
            .collect::<Vec<f32>>();

        // Create a tensor from the input
        let input = Tensor::from_vec(pixels, (1, pixel_count), &Device::Cpu).unwrap();

        // Create our first hidden layer where the dimensions are the number of total
        // pixels 28x28= 784 and the hidden layer dimension which we set to 128.
        let dense_weight = self
            .vb
            .get((pixel_count, self.hidden_dim), "d1_w")
            .unwrap()
            .transpose(0, 1)
            .unwrap();
        let dense_bias = self.vb.get(self.hidden_dim, "d1_b").unwrap();
        let dense = Linear::new(dense_weight, Some(dense_bias));

        // Create our output layer which is a dense layer with 128 inputs and 10 outputs
        let output_weight = self
            .vb
            .get((self.hidden_dim, digits_count), "d2_w")
            .unwrap()
            .transpose(0, 1)
            .unwrap();
        let output_bias = self.vb.get(digits_count, "d2_b").unwrap();
        let output = Linear::new(output_weight, Some(output_bias));

        // Perform the forward pass
        let next = dense.forward(&input).unwrap();
        let predictions = output.forward(&next).unwrap();
        predictions
            .argmax(1)
            .unwrap()
            .to_vec1::<u32>()
            .unwrap()
            .first()
            .cloned()
            .unwrap()
    }
}

/// To test the model, we have an example for each digit from 0-9. We iterate
/// over these examples and make sure the predicted digit matches the expected
/// digit.
#[test]
fn test_model() {
    let examples = vec![
        include_bytes!("../examples/example0.json").to_vec(),
        include_bytes!("../examples/example1.json").to_vec(),
        include_bytes!("../examples/example2.json").to_vec(),
        include_bytes!("../examples/example3.json").to_vec(),
        include_bytes!("../examples/example4.json").to_vec(),
        include_bytes!("../examples/example5.json").to_vec(),
        include_bytes!("../examples/example6.json").to_vec(),
        include_bytes!("../examples/example7.json").to_vec(),
        include_bytes!("../examples/example8.json").to_vec(),
        include_bytes!("../examples/example9.json").to_vec(),
    ];

    let model = Model::new(128);

    // Iterate over each example, parsing the JSON bytes into a vector of u32s
    // and then predicting the digit using that example.
    examples.iter().enumerate().for_each(|(i, e)| {
        let example: Vec<u32> = serde_json::from_slice(e.as_slice()).unwrap();
        assert_eq!(i as u32, model.predict(&example));
        println!("Successfully classified the digit {}", i);
    });
}
