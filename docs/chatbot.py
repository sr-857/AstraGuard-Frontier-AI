#importing necessary libraries
#importing necessary libraries
import psutil as p
import matplotlib.pyplot as plt
from datetime import datetime as tp
import flet as ft
import cv2 as c
import torch
from reportlab.lib.pagesizes import A4 
from ultralytics import YOLO#type: ignore
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib import colors
import torch.nn as nn
import math as m
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class MultiheadAttention(nn.Module):
    def __init__(self,d_model,num_heads,seq_len):
        super().__init__()
        self.d_model=d_model
        self.seq_len=seq_len
        self.num_heads=num_heads
        assert d_model % num_heads ==0 , "d_model must be divisible by num_heads"
        self.d_k=d_model//num_heads
        self.w_q=nn.Linear(d_model,d_model)
        self.w_k=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)
        self.w_o=nn.Linear(d_model,d_model)
    def scaled_dot_product(self,q,k,v,mask=None):
        # FIX: compute q @ k^T, THEN scale by sqrt(d_k)
        score = torch.matmul(q, k.transpose(-2, -1)) / m.sqrt(self.d_k)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(score, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output
    def split_heads(self,x):
        batch_size,seq_len,d_model=x.size()
        return x.view(batch_size,seq_len,self.num_heads,self.d_k).transpose(1,2)
    def combine_heads(self,x):
        batch_size,_,seq_len,_=x.size()
        return x.transpose(1,2).contiguous().view(batch_size,seq_len,self.d_model)
    def forward(self,q,k,v,mask=None):
        q=self.w_q(q)
        k=self.w_k(k)
        v=self.w_v(v)
        q=self.split_heads(q)
        k=self.split_heads(k)
        v=self.split_heads(v)
        scaled_attention=self.scaled_dot_product(q,k,v,mask)
        combined_attention=self.combine_heads(scaled_attention)
        output=self.w_o(combined_attention)
        return output
class Postional_encoding(nn.Module):
    def __init__(self,d_model,max_len=5000):
        super().__init__()
        pe=torch.zeros(max_len,d_model)
        position=torch.arange(0,max_len).unsqueeze(1)
        div_term=torch.exp(torch.arange(0,d_model,2)*(-m.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)
        pe=pe.unsqueeze(0)   # shape (1, max_len, d_model)
        self.register_buffer('pe',pe)
    def forward(self, x):
         return x + self.pe[:, :x.size(1)]#type:ignore
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, seq_len, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, num_heads, seq_len)
        # Feed-forward should map d_model -> d_ff -> d_model
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class Decoder_Layer(nn.Module):
    def __init__(self,d_model,num_heads,seq_len,ff_dim,dropout=0.1):
        super().__init__()
        self.mha1=MultiheadAttention(d_model,num_heads,seq_len)
        self.mha2=MultiheadAttention(d_model,num_heads,seq_len)
        self.ffn=nn.Sequential(
            nn.Linear(d_model,ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim,d_model)
        )
        self.layernorm1=nn.LayerNorm(d_model)
        self.layernorm2=nn.LayerNorm(d_model)
        self.layernorm3=nn.LayerNorm(d_model)
        self.dropout1=nn.Dropout(dropout)
        self.dropout2=nn.Dropout(dropout)
        self.dropout3=nn.Dropout(dropout)
    def forward(self,x,enc_output,src_mask=None,tgt_mask=None):
        attn1=self.mha1(x,x,x,tgt_mask)
        out1=self.layernorm1(x+self.dropout1(attn1))
        attn2=self.mha2(out1,enc_output,enc_output,src_mask)
        out2=self.layernorm2(out1+self.dropout2(attn2))
        ffn_output=self.ffn(out2)
        out3=self.layernorm3(out2+self.dropout3(ffn_output))
        return out3
class Transformer(nn.Module):
     def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = Postional_encoding(d_model, max_seq_length)

        # FIX: Encoder_Layer signature is (d_model, num_heads, seq_len, ff_dim, dropout)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, max_seq_length, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([Decoder_Layer(d_model, num_heads, max_seq_length, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

     def generate_mask(self, src, tgt):
        device = src.device  # Get the device from input tensor
        
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        
        seq_length = tgt.size(1)
        # Create nopeak_mask on the same device as src/tgt
        nopeak_mask = (1 - torch.triu(
            torch.ones(1, seq_length, seq_length, device=device),
            diagonal=1
        )).bool()
        
        # Both masks are now on the same device
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

     def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output

token=torch.load('Tokensizer.pt',weights_only=False)
# Move tokenized data to GPU
model = Transformer(
    src_vocab_size=len(token),
    tgt_vocab_size=len(token),
    d_model=512,
    num_heads=8,
    num_layers=6,   
    d_ff=2048,
    max_seq_length=128,
    dropout=0.1
).to(device)
model.load_state_dict(torch.load('model.pt',map_location=device))


class About:
    def __init__(self):
      super().__init__()
    def build(self):
        return ft.View(route='/About',controls=[ft.Text('ABOUT SECTION')])

class Home:
    def __init__(self):
        super().__init__()
    def build(self):
        return ft.View(route='/',controls=[ft.Text('DASHBOARD',size=40,color="#FFFFFFFF")])



#Battery class to get battery information
class Battery:
    def __init__(self):
        self.battery = p.sensors_battery()
    def get_battery_percentage(self):
        if self.battery is None:
            return "N/A"
        return self.battery.percent
    def is_plugged_in(self):
        if self.battery is None:
            return False
        return self.battery.power_plugged
    def get_time_left(self):
        if self.battery is None:
            return "Unknown"
        if self.is_plugged_in():
            return "Unlimited"
        hours = self.battery.secsleft // 3600
        minutes = (self.battery.secsleft % 3600) // 60
        return f"{hours}h {minutes}m"
class monitor:
    def __init__(self,query):
        self._query=query
# Load YOLO model
    @property
    def yolo_webcam(self):
        if self._query.lower()=='check':
            model = YOLO("yolov8x.pt")

            # Open webcam (0 = default camera)
            cap = c.VideoCapture(0)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # YOLO inference
                results = model(frame)

                # Show results
                annotated_frame = results[0].plot()
                c.imshow("YOLO Detection", annotated_frame)

                if c.waitKey(1) & 0xFF == ord('q'):
                    break
                cap.release()
                c.destroyAllWindows()

        else:
            print("Invalid query for yolo_webcam.")
        
            
    def detect_objects(self,d):
            l_1=[]
            model = YOLO("yolov8x.pt")
            cap = c.VideoCapture(0)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame)
                for r in results[0].boxes:
                    confidence = r.conf[0].item()
                    l_1.append(confidence)
                    
                if c.waitKey(1) & 0xFF == ord(d):
                    break
                cap.release()
                c.destroyAllWindows()
            return l_1


def predict(model, question, tokenizer, max_length=128, device='cuda'):
    model.eval()
    
    # Tokenize the question
    inputs = tokenizer(
        question,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)

    # Initialize target with start token
    tgt = torch.tensor([[tokenizer.cls_token_id]]).to(device)
    
    with torch.no_grad():#type:ignore
        for _ in range(max_length):
            # Get prediction
            output = model(inputs["input_ids"], tgt)
            
            # Get next token prediction
            next_token_logits = output[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
            
            # Add predicted token to target sequence
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # Stop if we predict the end token
            if next_token.item() == tokenizer.sep_token_id:
                break
    
    # Decode the generated answer
    answer = tokenizer.decode(tgt[0], skip_special_tokens=True)
    return answer

#ai chatbot interface complete

class ChatUI:
        def __init__(self):
            self.history = []
            self.input=[]
            battery=Battery()
            self.fg=battery.get_battery_percentage()
            self.dc=battery.get_time_left()
            self.dcv=battery.is_plugged_in() 
            self.gd=monitor('check').detect_objects(d='q')
            self.j=None
            for self.k in self.gd:
                if self.k>0.90:
                 self.j='Object detected re-orientation is began'
                else:
                    self.j='OBJECT IS NOT DETECTED SAFE'

        def create_pdf(self,path,e):
                    doc = SimpleDocTemplate(path)
                    data = [
                        ['Operations','Value'],
                        ['Battery percentage',f'BATTERY{self.fg}%'],
                        ['Battery CHARGING STATUS',f'Battery charging{self.dcv}'],
                        ['Battery Life',f'BATTTERY LIFE IS{self.dc}'],
                        ['Temperature','Temperature is 40'],
                        ['OBJECT DETECTION STATUS ',f'{self.j}']
                    ]
                    self.table = Table(data, colWidths=[200,220])
                    self.table.setStyle(TableStyle([
                        ("GRID", (0,0), (-1,-1), 1, colors.green),
                        ("BACKGROUND", (0,0), (-1,0), colors.yellow),
                        ("ALIGN", (0,0), (-1,-1), "CENTER")
                    ]))

                    doc.build([
                        Paragraph("ASTRAGUARD AI REPORT"),
                        self.table
                    ])
                    
        def graph(self,e):
            q=str(self.user_input.value)
            if 'battery temperature vs time'in q.lower():
                temperature=[20,30,40,50,40,100]
                time=[1,2,3,4,5,6]
                plt.plot(time,temperature,label='BATTERY TEMPERATURE VS TIME')
                plt.legend(fontsize=12,title='ASTRAGUARD AI',loc='upper left',shadow=True)
                return plt.savefig('Graph.png')
            elif 'temperature vs time'in q.lower():
                temperature=[20,30,40,50,40,100]
                time=[1,2,3,4,5,6]
                plt.plot(time,temperature,label='TEMPERATURE VS TIME')
                plt.legend(fontsize=12,title='ASTRAGUARD AI',loc='upper left',shadow=True)
                return plt.savefig('Graph.png'),plt.close()
        
        def send(self, e):
                q=self.user_input.value
                self.a=predict(model,q,token)
                self.history.append(q)
                self.history.append(self.a)
                return (
                    e.page.add(ft.Text(f'Q:\n{q}')),
                    e.page.add(ft.Text(f'A:\n{self.a}')),
                    e.page.update()
                )
        def generate_btn(self,e):
            if 'pdf' in str(self.user_input.value).lower():
               return self.create_pdf('REPORT.pdf',e)
            elif 'temperature vs time' in str(self.user_input.value).lower():
                return self.graph(e)
            e.page.update()
            
        def toggle_btn(self,e):
            text=str(self.user_input.value).lower()
            if 'pdf' in text or 'temperature vs time' in text:
                self.generate.visible=True
            else:
                self.generate.visible=False
            e.page.update()
            
        def view(self):
            self.user_input = ft.TextField(
                hint_text="Ask ASTRAGUARD AI...",
                expand=True,
                enable_interactive_selection=True,
                adaptive=True,
                autofocus=True,
                on_submit=self.send,
                on_change=self.toggle_btn,color="#9BB03C"
            )
            self.input.append(self.user_input.value)
            return self.user_input
        def buttons(self):
            self.generate=ft.ElevatedButton('GENERATED',on_click=self.generate_btn,visible=False,bgcolor="#6F922F")
            button=ft.View(route='/chatui',controls=[ft.Row([self.view(),ft.ElevatedButton('Send',on_click=lambda e:self.send(e),bgcolor="#39922F")]),self.generate])
            return button

class Myapp:
    def __init__(self,page:ft.Page):
        self.page=page
        self.page.title = "ASTRAGUARD AI"
        self.page.window.icon="logo.png"
        self.page.on_route_change = self.route_change
        # Bottom Navigation
        self.page.bottom_appbar = ft.BottomAppBar(
            content=ft.Row(
                alignment=ft.MainAxisAlignment.SPACE_AROUND,
                controls=[
                    ft.IconButton(
                        icon=ft.Icons.HOME,
                        on_click=lambda e: self.page.go("/")
                    ),
                    ft.IconButton(
                        icon=ft.Icons.CHAT,
                        on_click=lambda e: self.page.go("/chatui")
                    ),
                    ft.IconButton(
                        icon=ft.Icons.INFO,
                        on_click=lambda e: self.page.go("/About")
                    ),
                ]
            )
        )

        self.page.go("/")
    def route_change(self):
        self.page.views.clear()

        if self.page.route == "/":
            self.page.views.append(Home().build())

        elif self.page.route == "/chatui":
            self.page.views.append(ChatUI().buttons())

        elif self.page.route == "/About":
            self.page.views.append(About().build())

        self.page.update()
def main(page:ft.Page):
    Myapp(page)


ft.run(main)
