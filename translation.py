import tensorflow as tf
from transformer import Transformer

# training data

# Create a dictionary containing English sentences
# and their Hindi translations
translation_data = {
    "I like tea.": "मुझे चाय पसंद है।",
    "I like coffee.": "मुझे कॉफी पसंद है।",
    "I drink water every morning.": "मैं हर सुबह पानी पीता हूँ।",
    "She likes to read books.": "उसे किताबें पढ़ना पसंद है।",
    "He plays football every Sunday.": "वह हर रविवार फुटबॉल खेलता है।",
    "They go to school together.": "वे साथ में स्कूल जाते हैं।",
    "We live in a small city.": "हम एक छोटे शहर में रहते हैं।",
    "My brother is a student.": "मेरा भाई एक छात्र है।",
    "My sister likes music.": "मेरी बहन को संगीत पसंद है।",
    "The dog is sleeping.": "कुत्ता सो रहा है।",
    "The cat is sitting on the chair.": "बिल्ली कुर्सी पर बैठी है।",
    "I have a new phone.": "मेरे पास एक नया फोन है।",
    "She has a beautiful dress.": "उसके पास एक सुंदर पोशाक है।",
    "He bought a new car.": "उसने एक नई कार खरीदी।",
    "We watched a movie yesterday.": "हमने कल एक फिल्म देखी।",
    "They visited their grandparents.": "वे अपने दादा-दादी से मिलने गए।",
    "I went to the market.": "मैं बाजार गया।",
    "She cooked dinner for us.": "उसने हमारे लिए रात का खाना बनाया।",
    "He opened the window.": "उसने खिड़की खोली।",
    "The baby is crying.": "बच्चा रो रहा है।",
    "I wake up early every day.": "मैं हर दिन जल्दी उठता हूँ।",
    "She goes to work by bus.": "वह बस से काम पर जाती है।",
    "He walks to school.": "वह पैदल स्कूल जाता है।",
    "We eat breakfast together.": "हम साथ में नाश्ता करते हैं।",
    "They play cricket in the evening.": "वे शाम को क्रिकेट खेलते हैं।",
    "I am learning English.": "मैं अंग्रेजी सीख रहा हूँ।",
    "She is learning Python.": "वह पायथन सीख रही है।",
    "He is watching television.": "वह टेलीविजन देख रहा है।",
    "We are studying machine learning.": "हम मशीन लर्निंग पढ़ रहे हैं।",
    "They are playing in the garden.": "वे बगीचे में खेल रहे हैं।",
    "I finished my homework.": "मैंने अपना गृहकार्य पूरा किया।",
    "She cleaned her room.": "उसने अपना कमरा साफ किया।",
    "He called his friend.": "उसने अपने दोस्त को फोन किया।",
    "We visited the museum.": "हम संग्रहालय गए।",
    "They enjoyed the party.": "उन्होंने पार्टी का आनंद लिया।",
    "I saw a beautiful bird.": "मैंने एक सुंदर पक्षी देखा।",
    "She found her lost book.": "उसे अपनी खोई हुई किताब मिल गई।",
    "He forgot his keys.": "वह अपनी चाबियाँ भूल गया।",
    "We missed the train.": "हमारी ट्रेन छूट गई।",
    "They reached home late.": "वे देर से घर पहुँचे।",
    "The sun is shining.": "सूरज चमक रहा है।",
    "The sky is blue.": "आसमान नीला है।",
    "It is raining outside.": "बाहर बारिश हो रही है।",
    "The weather is very pleasant.": "मौसम बहुत सुहावना है।",
    "The wind is blowing strongly.": "हवा तेज चल रही है।",
    "The flowers are beautiful.": "फूल सुंदर हैं।",
    "The garden looks green.": "बगीचा हरा दिखता है।",
    "The river is very clean.": "नदी बहुत साफ है।",
    "The mountain is covered with snow.": "पहाड़ बर्फ से ढका हुआ है।",
    "The road is busy today.": "आज सड़क व्यस्त है।",
    "I need some water.": "मुझे कुछ पानी चाहिए।",
    "She wants a cup of tea.": "उसे एक कप चाय चाहिए।",
    "He needs a new bag.": "उसे एक नए बैग की जरूरत है।",
    "We want to learn more.": "हम और सीखना चाहते हैं।",
    "They need some help.": "उन्हें कुछ मदद चाहिए।",
    "I have a question.": "मेरे पास एक सवाल है।",
    "She knows the answer.": "वह जवाब जानती है।",
    "He does not understand the problem.": "वह समस्या को नहीं समझता है।",
    "We understand the situation.": "हम स्थिति को समझते हैं।",
    "They know my name.": "वे मेरा नाम जानते हैं।",
    "Where do you live?": "तुम कहाँ रहते हो?",
    "What are you doing?": "तुम क्या कर रहे हो?",
    "Why are you late?": "तुम देर से क्यों आए?",
    "When will you come?": "तुम कब आओगे?",
    "Who is your teacher?": "तुम्हारे शिक्षक कौन हैं?",
    "Which book do you want?": "तुम्हें कौन सी किताब चाहिए?",
    "How are you today?": "आज तुम कैसे हो?",
    "Where is my phone?": "मेरा फोन कहाँ है?",
    "What time is the meeting?": "बैठक कितने बजे है?",
    "Why is the door open?": "दरवाजा खुला क्यों है?",
    "Please open the door.": "कृपया दरवाजा खोलो।",
    "Please close the window.": "कृपया खिड़की बंद करो।",
    "Give me some water.": "मुझे थोड़ा पानी दो।",
    "Come with me.": "मेरे साथ आओ।",
    "Wait for me.": "मेरा इंतजार करो।",
    "Listen to the teacher.": "शिक्षक की बात सुनो।",
    "Read this book.": "यह किताब पढ़ो।",
    "Write your name here.": "अपना नाम यहाँ लिखो।",
    "Please help me.": "कृपया मेरी मदद करो।",
    "Take this bag.": "यह बैग लो।",
    "I am happy today.": "मैं आज खुश हूँ।",
    "She is very kind.": "वह बहुत दयालु है।",
    "He is a good person.": "वह एक अच्छा व्यक्ति है।",
    "We are ready for the test.": "हम परीक्षा के लिए तैयार हैं।",
    "They are very excited.": "वे बहुत उत्साहित हैं।",
    "I am feeling tired.": "मैं थका हुआ महसूस कर रहा हूँ।",
    "She looks happy.": "वह खुश दिखती है।",
    "He seems worried.": "वह चिंतित लगता है।",
    "We are proud of our team.": "हमें अपनी टीम पर गर्व है।",
    "They are afraid of the dark.": "वे अंधेरे से डरते हैं।",
    "My father works in an office.": "मेरे पिता एक कार्यालय में काम करते हैं।",
    "My mother cooks delicious food.": "मेरी माँ स्वादिष्ट खाना बनाती हैं।",
    "My brother plays the guitar.": "मेरा भाई गिटार बजाता है।",
    "My sister studies at college.": "मेरी बहन कॉलेज में पढ़ती है।",
    "My parents live in the village.": "मेरे माता-पिता गाँव में रहते हैं।",
    "My friend has a new laptop.": "मेरे दोस्त के पास एक नया लैपटॉप है।",
    "Our teacher explains everything clearly.": "हमारे शिक्षक सब कुछ स्पष्ट रूप से समझाते हैं।",
    "His father drives a bus.": "उसके पिता बस चलाते हैं।",
    "Her mother works at a hospital.": "उसकी माँ अस्पताल में काम करती हैं।",
    "Their house is near the river.": "उनका घर नदी के पास है।",
    "I bought some vegetables.": "मैंने कुछ सब्जियाँ खरीदीं।",
    "She bought a new notebook.": "उसने एक नई नोटबुक खरीदी।",
    "He ordered a pizza.": "उसने पिज्जा ऑर्डर किया।",
    "We made a cake.": "हमने एक केक बनाया।",
    "They prepared lunch.": "उन्होंने दोपहर का खाना तैयार किया।",
    "I washed my clothes.": "मैंने अपने कपड़े धोए।",
    "She cleaned the kitchen.": "उसने रसोई साफ की।",
    "He repaired the bicycle.": "उसने साइकिल की मरम्मत की।",
    "We painted the wall.": "हमने दीवार को रंगा।",
    "They built a small house.": "उन्होंने एक छोटा घर बनाया।",
    "I will call you tomorrow.": "मैं तुम्हें कल फोन करूँगा।",
    "She will visit us next week.": "वह अगले सप्ताह हमसे मिलने आएगी।",
    "He will start a new job.": "वह एक नई नौकरी शुरू करेगा।",
    "We will meet in the evening.": "हम शाम को मिलेंगे।",
    "They will travel next month.": "वे अगले महीने यात्रा करेंगे।",
    "I am going to the market.": "मैं बाजार जा रहा हूँ।",
    "She is going to buy groceries.": "वह किराने का सामान खरीदने जा रही है।",
    "He is going to play cricket.": "वह क्रिकेट खेलने जा रहा है।",
    "We are going to watch a movie.": "हम एक फिल्म देखने जा रहे हैं।",
    "They are going to visit Delhi.": "वे दिल्ली जाने वाले हैं।",
    "I can speak English.": "मैं अंग्रेजी बोल सकता हूँ।",
    "She can drive a car.": "वह कार चला सकती है।",
    "He can solve this problem.": "वह इस समस्या को हल कर सकता है।",
    "We can finish the work today.": "हम आज काम पूरा कर सकते हैं।",
    "They can help you.": "वे आपकी मदद कर सकते हैं।",
    "I cannot find my wallet.": "मुझे अपना बटुआ नहीं मिल रहा है।",
    "She cannot come today.": "वह आज नहीं आ सकती।",
    "He cannot lift the box.": "वह डिब्बा नहीं उठा सकता।",
    "We cannot wait any longer.": "हम और इंतजार नहीं कर सकते।",
    "They cannot understand Hindi.": "वे हिंदी नहीं समझ सकते।",
    "I like this book because it is interesting.": "मुझे यह किताब पसंद है क्योंकि यह दिलचस्प है।",
    "She stayed home because she was sick.": "वह घर पर रही क्योंकि वह बीमार थी।",
    "He went outside because the weather was nice.": "वह बाहर गया क्योंकि मौसम अच्छा था।",
    "We studied hard because the exam was near.": "हमने कठिन अध्ययन किया क्योंकि परीक्षा पास थी।",
    "They smiled because they were happy.": "वे मुस्कुराए क्योंकि वे खुश थे।",
    "I opened the door when I heard a noise.": "जब मैंने आवाज सुनी तो मैंने दरवाजा खोला।",
    "She called me when she reached home.": "जब वह घर पहुँची तो उसने मुझे फोन किया।",
    "He sleeps early because he works in the morning.": "वह जल्दी सोता है क्योंकि वह सुबह काम करता है।",
    "We stayed inside because it was raining.": "हम अंदर रहे क्योंकि बारिश हो रही थी।",
    "They stopped playing because it became dark.": "उन्होंने खेलना बंद कर दिया क्योंकि अंधेरा हो गया।",
    "The boy is reading a book.": "लड़का एक किताब पढ़ रहा है।",
    "The girl is drawing a picture.": "लड़की एक तस्वीर बना रही है।",
    "The teacher is writing on the board.": "शिक्षक बोर्ड पर लिख रहे हैं।",
    "The students are listening carefully.": "छात्र ध्यान से सुन रहे हैं।",
    "The children are playing outside.": "बच्चे बाहर खेल रहे हैं।",
    "The man is riding a bicycle.": "आदमी साइकिल चला रहा है।",
    "The woman is carrying a bag.": "महिला एक बैग ले जा रही है।",
    "The farmer is working in the field.": "किसान खेत में काम कर रहा है।",
    "The doctor is talking to the patient.": "डॉक्टर मरीज से बात कर रहे हैं।",
    "The driver is waiting for the passengers.": "ड्राइवर यात्रियों का इंतजार कर रहा है।",
    "I have been waiting for an hour.": "मैं एक घंटे से इंतजार कर रहा हूँ।",
    "She has finished her work.": "उसने अपना काम पूरा कर लिया है।",
    "He has already eaten lunch.": "वह पहले ही दोपहर का खाना खा चुका है।",
    "We have visited this place before.": "हम पहले इस जगह पर जा चुके हैं।",
    "They have completed the project.": "उन्होंने परियोजना पूरी कर ली है।",
    "I had finished my homework before dinner.": "मैंने रात के खाने से पहले अपना गृहकार्य पूरा कर लिया था।",
    "She had left before I arrived.": "मेरे पहुँचने से पहले वह जा चुकी थी।",
    "He had already seen the movie.": "वह पहले ही फिल्म देख चुका था।",
    "We had reached the station early.": "हम स्टेशन पर जल्दी पहुँच गए थे।",
    "They had completed the task before the deadline.": "उन्होंने समय सीमा से पहले कार्य पूरा कर लिया था।",
    "The computer is on the table.": "कंप्यूटर मेज पर है।",
    "My laptop is very fast.": "मेरा लैपटॉप बहुत तेज है।",
    "This phone has a good camera.": "इस फोन में अच्छा कैमरा है।",
    "The internet is not working.": "इंटरनेट काम नहीं कर रहा है।",
    "I use Python for programming.": "मैं प्रोग्रामिंग के लिए पायथन का उपयोग करता हूँ।",
    "She is learning artificial intelligence.": "वह कृत्रिम बुद्धिमत्ता सीख रही है।",
    "He works with machine learning models.": "वह मशीन लर्निंग मॉडल के साथ काम करता है।",
    "We are building a neural network.": "हम एक न्यूरल नेटवर्क बना रहे हैं।",
    "They are training a computer vision model.": "वे एक कंप्यूटर विजन मॉडल को प्रशिक्षित कर रहे हैं।",
    "The model gives good results.": "मॉडल अच्छे परिणाम देता है।",
    "I want to become a machine learning engineer.": "मैं मशीन लर्निंग इंजीनियर बनना चाहता हूँ।",
    "She wants to learn data science.": "वह डेटा साइंस सीखना चाहती है।",
    "He studies deep learning every day.": "वह हर दिन डीप लर्निंग पढ़ता है।",
    "We use TensorFlow for the project.": "हम परियोजना के लिए TensorFlow का उपयोग करते हैं।",
    "They are working on a translation system.": "वे एक अनुवाद प्रणाली पर काम कर रहे हैं।",
    "The model understands simple sentences.": "मॉडल सरल वाक्यों को समझता है।",
    "I trained the model yesterday.": "मैंने कल मॉडल को प्रशिक्षित किया।",
    "She tested the model today.": "उसने आज मॉडल का परीक्षण किया।",
    "He saved the trained model.": "उसने प्रशिक्षित मॉडल को सहेजा।",
    "We loaded the model again.": "हमने मॉडल को फिर से लोड किया।",
    "I like sunny days.": "मुझे धूप वाले दिन पसंद हैं।",
    "She enjoys listening to music.": "उसे संगीत सुनना पसंद है।",
    "He loves playing football.": "उसे फुटबॉल खेलना बहुत पसंद है।",
    "We enjoy spending time together.": "हमें साथ समय बिताना पसंद है।",
    "They like visiting new places.": "उन्हें नई जगहों पर जाना पसंद है।",
    "I want to travel to another country.": "मैं किसी दूसरे देश की यात्रा करना चाहता हूँ।",
    "She wants to learn a new language.": "वह एक नई भाषा सीखना चाहती है।",
    "He wants to start his own business.": "वह अपना खुद का व्यवसाय शुरू करना चाहता है।",
    "We hope to see you soon.": "हमें उम्मीद है कि हम आपको जल्द देखेंगे।",
    "They plan to visit their friends.": "वे अपने दोस्तों से मिलने की योजना बना रहे हैं।",
    "Today is a beautiful day.": "आज एक सुंदर दिन है।",
    "Tomorrow will be a busy day.": "कल एक व्यस्त दिन होगा।",
    "Yesterday was very difficult.": "कल बहुत कठिन दिन था।",
    "The meeting starts at ten o'clock.": "बैठक दस बजे शुरू होती है।",
    "The train arrives at six o'clock.": "ट्रेन छह बजे पहुँचती है।",
    "My class starts in the afternoon.": "मेरी कक्षा दोपहर में शुरू होती है।",
    "We usually have dinner at eight.": "हम आमतौर पर आठ बजे रात का खाना खाते हैं।",
    "She goes to bed early.": "वह जल्दी सोने जाती है।",
    "He wakes up at seven every morning.": "वह हर सुबह सात बजे उठता है।",
    "I am happy to learn something new.": "मैं कुछ नया सीखकर खुश हूँ।"
}


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
    num_heads=2,
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



