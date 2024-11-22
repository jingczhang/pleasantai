from gradio_client import Client, handle_file

client = Client("jingczhang/scorpio")
image_url = 'images/sanxingdui.jpg'

result = client.predict(
                img=handle_file(image_url),
                        api_name="/predict"
                        )
print(result)

what = result['label']
probs = 100 
for item in result['confidences']:
    if item['label'] == what:
        probs = item['confidence']
print(f"This is: {what}.")
print(f"Probability it's {what}: {probs}")
