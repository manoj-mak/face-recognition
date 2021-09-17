import face_recognition
from PIL import Image, ImageDraw

image_of_indira = face_recognition.load_image_file('./img/known/ind4.jpg')
indira_face_encoding = face_recognition.face_encodings(image_of_indira)[0]

image_of_Nehru = face_recognition.load_image_file('./img/known/n2.jpg')
Nehru_face_encoding = face_recognition.face_encodings(image_of_Nehru)[0]

image_of_Gandhi = face_recognition.load_image_file('./img/known/Gandhi.jpg')
Gandhi_face_encoding = face_recognition.face_encodings(image_of_Gandhi)[0]

image_of_lal = face_recognition.load_image_file('./img/known/lal2.jpeg')
lal_face_encoding = face_recognition.face_encodings(image_of_lal)[0]





image_of_radha = face_recognition.load_image_file('./img/known/radha1.jpg')
radha_face_encoding = face_recognition.face_encodings(image_of_radha)[0]

image_of_kalam = face_recognition.load_image_file('./img/known/k1.jpg')
kalam_face_encoding = face_recognition.face_encodings(image_of_kalam)[0]


#  Create arrays of encodings and names
known_face_encodings = [
  indira_face_encoding,
  Nehru_face_encoding,
  Gandhi_face_encoding,
  lal_face_encoding,
  radha_face_encoding,
  kalam_face_encoding
]

known_face_names = [
  "Indira",
  "Nehru",
  "Gandhi",
  "Lal bahadur shastri",
  "sp radhakrishnan",
  "Adbul Kalam"
]

# Load test image to find faces in
test_image = face_recognition.load_image_file('./img/unknown/sample2.jpg')

# Find faces in test image
face_locations = face_recognition.face_locations(test_image)
face_encodings = face_recognition.face_encodings(test_image, face_locations)

# Convert to PIL format
pil_image = Image.fromarray(test_image)

# Create a ImageDraw instance
draw = ImageDraw.Draw(pil_image)

# Loop through faces in test image
for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
  matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

  name = "Unknown Person"

  # If match
  if True in matches:
    first_match_index = matches.index(True)
    name = known_face_names[first_match_index]
  
  # Draw box
  draw.rectangle(((left, top), (right, bottom)), outline=(255,255,0))

  # Draw label
  text_width, text_height = draw.textsize(name)
  draw.rectangle(((left,bottom - text_height - 10), (right, bottom)), fill=(255,255,0), outline=(255,255,0))
  draw.text((left + 20, bottom - text_height - 5), name, fill=(0,0,0))

del draw

# Display image
pil_image.show()

# Save image
pil_image.save('identity.jpg')
