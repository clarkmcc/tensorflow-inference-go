# Running TensorFlow Models Natively in Go
This project is an example of building and training a Tensorflow model, manually porting it to Rust using [candle](https://github.com/huggingface/candle), compiling the Rust to WebAssembly, and running the WebAssembly module from Go using [Scale](https://scale.sh/).

The following files are included:
* [**TensorFlow Model**](./model-notebook.ipynb) included as a Jupyter notebook
* [**Rust Model**](./model/lib.rs) ported from the TensorFlow model using candle
* [**Go Binary**](./main.go) that loads the Rust model and runs it on a test input

## Running the Example
```shell
$ go run main.go
Successfully classified the digit 0 in 1.357ms
Successfully classified the digit 1 in 618.541µs
Successfully classified the digit 2 in 607.417µs
Successfully classified the digit 3 in 636.5µs
Successfully classified the digit 4 in 622.291µs
Successfully classified the digit 5 in 638.375µs
Successfully classified the digit 6 in 666.25µs
Successfully classified the digit 7 in 647.791µs
Successfully classified the digit 8 in 614.666µs
Successfully classified the digit 9 in 608.333µs
```

## Developing
Make sure you have [Scale](https://scale.sh/docs/cli/installation) installed.
```shell
$ curl -fsSL https://dl.scale.sh | sh
```

Any changes to the Rust code, or the signature need to be re-compiled to WebAssembly. Run the following to re-generate all the signatures and compile the Rust code to WebAssembly.
```shell
$ make generate
``` 