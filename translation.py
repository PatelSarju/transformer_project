import tensorflow as tf
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu
from transformer import Transformer

# training data

# Create a dictionary containing English sentences
# and their Hindi translations
data = pd.read_csv("english_hindi_sentence_pairs.csv")
translation_data = dict(zip(data["english"], data["hindi"]))

# seperate english and hindi sentences

# Get all English sentences from the dictionary
english_sentences = list(
    translation_data.keys()
)

# Get all Hindi translations from the dictionary
hindi_sentences = [
    "<start> " + sentence + " <end>"
    for sentence in translation_data.values()
]

# create english tokenizer

# Create English tokenizer
english_tokenizer = tf.keras.preprocessing.text.Tokenizer()

# Build English vocabulary using all English sentences
english_tokenizer.fit_on_texts(
    english_sentences
)

# Convert all English sentences into token IDs
english_tokens = (
    english_tokenizer
    .texts_to_sequences(english_sentences)
)

# create hindi tokenizer

# Create Hindi tokenizer
# filters="" keeps <start> and <end>
hindi_tokenizer = tf.keras.preprocessing.text.Tokenizer(
    filters=""
)

# Build Hindi vocabulary using all Hindi sentences
hindi_tokenizer.fit_on_texts(
    hindi_sentences
)

# Convert all Hindi sentences into token IDs
hindi_tokens = (hindi_tokenizer.texts_to_sequences(hindi_sentences))

# create decoder input and target

# Store decoder inputs for all sentences
decoder_inputs = []

# Store targets for all sentences
targets = []


# Process every Hindi sentence
for tokens in hindi_tokens:

    # Decoder receives everything except <end>
    decoder_input = tokens[:-1]

    # Target contains everything except <start>
    target = tokens[1:]

    # Add decoder input to the list
    decoder_inputs.append(decoder_input)

    # Add target to the list
    targets.append(target)

# pad the sequences

# Pad English sentences so they have the same length
english_tokens = tf.keras.preprocessing.sequence.pad_sequences(
    english_tokens,
    padding="post"
)

# Pad Decoder inputs so they have the same length
decoder_inputs = tf.keras.preprocessing.sequence.pad_sequences(
    decoder_inputs,
    padding="post"
)

# Pad targets so they have the same length
targets = tf.keras.preprocessing.sequence.pad_sequences(
    targets,
    padding="post"
)

# convert inputs into tensors

# Convert English token IDs into TensorFlow tensor
english_input = tf.constant(
    english_tokens,
    dtype=tf.int32
)

# Convert Decoder input IDs into TensorFlow tensor
decoder_input_tensor = tf.constant(
    decoder_inputs,
    dtype=tf.int32
)

# Convert target IDs into TensorFlow tensor
target_tensor = tf.constant(
    targets,
    dtype=tf.int32
)

# get vocabulary sizes

# Get English vocabulary size
english_vocab_size = (
    max(english_tokenizer.word_index.values()) + 1
)

# Get Hindi vocabulary size
hindi_vocab_size = (
    max(hindi_tokenizer.word_index.values()) + 1
)

# create transformer

# Create the Transformer model
transformer = Transformer(
    english_vocab_size=english_vocab_size,
    hindi_vocab_size=hindi_vocab_size,
    d_model=128,
    num_heads=4,
    dff=512
)

# create loss function

# Create Sparse Categorical Cross-Entropy loss
loss_function = (
    tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction="none"
    )
)


# Calculate loss while ignoring padding tokens
def calculate_loss(real, predicted):

    # Calculate loss for every token
    loss = loss_function(
        real,
        predicted
    )

    # Create mask for non-padding tokens
    mask = tf.cast(
        tf.not_equal(real, 0),
        tf.float32
    )

    # Ignore padding token loss
    loss = loss * mask

    # Calculate average loss
    return (
        tf.reduce_sum(loss) /
        tf.reduce_sum(mask)
    )

# create optimizer

# Create Adam optimizer
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001
)

# train the transformer

# Number of training iterations
epochs = 400

# Repeat training multiple times
for epoch in range(epochs):

    # Record operations for gradient calculation
    with tf.GradientTape() as tape:

        # Pass all English sentences and
        # all Decoder inputs through Transformer
        logits = transformer(
            english_input,
            decoder_input_tensor
        )

        # Calculate loss for all sentences
        loss = calculate_loss(
            target_tensor,
            logits
        )

    # Calculate gradients
    gradients = tape.gradient(
        loss,
        transformer.trainable_variables
    )

    # Update Transformer weights
    optimizer.apply_gradients(
        zip(
            gradients,
            transformer.trainable_variables
        )
    )

    # Print loss every 50 epochs
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}, "f"Loss: {loss.numpy():.4f}")

# take the user input

# Ask the user for an English sentence
user_sentence = input("\nEnter an English sentence: ")

# tokenize user input

# Convert the user's English sentence
# into token IDs
user_tokens = (
    english_tokenizer
    .texts_to_sequences(
        [user_sentence]
    )
)

# Convert the token IDs into a TensorFlow tensor
user_input = tf.constant(
    user_tokens,
    dtype=tf.int32
)

# get start and end token

# Get <start> token ID
start_token = (hindi_tokenizer.word_index["<start>"])

# Get <end> token ID
end_token = (hindi_tokenizer.word_index["<end>"])

# start the translation

# Start generation with <start>
generated_tokens = [start_token]

# generate hindi tokens

# Generate Hindi tokens one by one
for _ in range(20):

    # Convert generated tokens into tensor
    decoder_input = tf.constant(
        [generated_tokens],
        dtype=tf.int32
    )

    # Pass user input and generated
    # Hindi tokens through Transformer
    logits = transformer(
        user_input,
        decoder_input
    )

    # Take prediction from the last position
    next_token_logits = (
        logits[:, -1, :]
    )

    # Select token with highest probability
    next_token = tf.argmax(
        next_token_logits,
        axis=-1
    ).numpy()[0]

    # Add predicted token
    generated_tokens.append(int(next_token))

    # Stop when <end> is generated
    if next_token == end_token:
        break

# convert token ID into words

# Convert generated token IDs into Hindi words
generated_words = [
    hindi_tokenizer.index_word[token]
    for token in generated_tokens
]

# create final translation

# Remove <start> and <end> from the final output
generated_words = generated_words[1:-1]


# Create final Hindi sentence
translation = " ".join(generated_words)

# print translation

print("Generated translation:",translation)


# calculate BLEU score

# Ask for the correct Hindi translation
reference_sentence = input(
    "Enter the correct Hindi translation: "
)

# Convert reference Hindi sentence into words
reference_words = reference_sentence.split()

# Convert generated Hindi translation into words
generated_words = translation.split()

# Calculate BLEU score
bleu_score = corpus_bleu(
    [[reference_words]],
    [generated_words]
)

# Print BLEU score
print(
    f"BLEU Score: {bleu_score:.4f}"
)
