const {pipeline} = App;
const {prompt  } = App.instance.state;
const maxLength = pipeline.tokenizer.model_max_length;
const tokens = pipeline.tokenizer(
    prompt,
    {
      return_tensor: false,
      padding: false,
      max_length: maxLength,
      return_tensor_dtype: 'int32',
    },
);
console.log(tokens);
