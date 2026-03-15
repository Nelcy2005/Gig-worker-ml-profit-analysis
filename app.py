import gradio as gr

def profit_calculator(earnings,petrol,penalty):

    net = earnings - petrol - penalty

    rate = net/earnings

    if net>0:
        status="PROFIT"
    else:
        status="LOSS"

    return f"""
Net Income: {net}

Profit Rate: {rate:.2f}

Status: {status}
"""

demo = gr.Interface(
    fn=profit_calculator,

    inputs=[
        gr.Number(label="Earnings"),
        gr.Number(label="Petrol Cost"),
        gr.Number(label="Penalty")
    ],

    outputs="text",

    title="Delivery Worker Profit Analyzer"
)

demo.launch()