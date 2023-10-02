import gradio as gr

def greet(name):
    return "Hellooooooooooooo " + name + "!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
    
demo.launch()   