import tkinter as tk
import tkinter.ttk as ttk
import tensorflow as tf

import sys
import os
import shutil
sys.path.append(os.path.abspath('./'))
from helper import load_hyper_params, initialize_model_from_local, plot_attention
from evaluate import evaluate
from Vocabulary import Vocabulary


class AppWindow(object):
    initial_width = 465
    initial_height = 645

    hparams_path = './checkpoints/hparams.json'
    ckpt_path = './checkpoints/checkpoints-125ep-3/'
    ckpt_number = 'ckpt-5'
    #ckpt_path = './checkpoints/checkpoints-final-2/'
    #ckpt_number = 'ckpt-2'
    #vocab_path = './vocab/vocabulary_no_ap_indexed.db'
    vocab_path = './vocab/full_vocab_validated.db'

    def __init__(self):
        self.text_history = []
        self.attention_weights = None
        self.input = None
        self.output = None
        self.mode = 'chat'
        self.auto_inp = None

        self.root = tk.Tk()
        self.root.style = ttk.Style()
        self.root.style.theme_use('clam')

        self.root.iconphoto(True, tk.PhotoImage(file='./GUI/icon.png'))
        self.root.title('Seq2Seq Chatbot')
        self.root.geometry(str(self.initial_width)+'x'+str(self.initial_height))
        self.root.resizable(False, False)
        self.root.bind('<Return>', self.get_result)
        
        chat_frame = ttk.Frame(self.root, width=self.initial_width, height=self.initial_height)  # , bg='blue')
        chat_frame.pack()
        inp_frame = ttk.Frame(self.root, width=self.initial_width, height=50)  # , bg='red')
        inp_frame.pack()
        att_frame = ttk.Frame(self.root, width=self.initial_width, height=25)  # , bg='green')
        att_frame.pack()
        

        self.scrollbar = ttk.Scrollbar(chat_frame)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.display_text = tk.Text(chat_frame, yscrollcommand=self.scrollbar, font=('Helvetica', 10), width=60, height=33, bd=1, relief='solid', padx=5, pady=10)
        self.display_text.pack(padx=5, pady=5)
        self.scrollbar.config(command=self.display_text.yview)

        self.entry = tk.StringVar()
        self.entry_w = ttk.Entry(inp_frame, state=tk.DISABLED, width=45, textvariable=self.entry)
        self.entry_w.pack(side=tk.LEFT, padx=4)
        self.main_button = ttk.Button(inp_frame, text='Send', command=lambda: self.get_result(None), width=5)
        self.main_button.pack(side=tk.LEFT, padx=8, pady=4)

        self.label_text = tk.StringVar()
        tk.Label(att_frame, width=20, textvariable=self.label_text, anchor=tk.W).pack(side=tk.LEFT)
        self.auto_button = ttk.Button(att_frame, text='Auto Chat', command=self.auto_button_fnc, width=10)
        self.auto_button.pack(side=tk.RIGHT, padx=3)
        att_buton = ttk.Button(att_frame, text='Show Attention', command=self.attention_button_fnc)
        att_buton.pack(side=tk.RIGHT, padx=3)
        self.reset_button = ttk.Button(att_frame, text='Reset', command=self.reset_button_fnc)
        self.reset_button.pack(side=tk.RIGHT)
        self.reset_button.pack_forget()

        self.display_text.config(state=tk.NORMAL)
        self.display_text.insert(tk.END,'Loading model...')
        self.display_text.config(state=tk.DISABLED)
        self.root.update_idletasks()
        self.root.update()


    def load_model(self):
        self.hparams = load_hyper_params(self.hparams_path)
        self.v, self.enc, self.dec, self.opt = initialize_model_from_local(self.vocab_path, self.hparams, de_tokenize=False, verbose=False)
        checkpoint = tf.train.Checkpoint(optimizer=self.opt, encoder=self.enc, decoder=self.dec)
        checkpoint.restore(self.ckpt_path + self.ckpt_number).expect_partial()
        _, _, _ = evaluate(u'test', self.v, self.enc, self.dec, self.hparams['MAX_LEN'])

    
    def get_result(self, event=None):
        inp = self.entry.get()
        if self.mode == 'chat':
            if inp != '':
                self.entry.set('')
                self.output, self.input, attention_plot = evaluate(inp, self.v, self.enc, self.dec, self.hparams['MAX_LEN'])
                self.attention_weights = attention_plot[:len(self.output.split(' ')), :len(self.input.split(' '))]
                
                res = Vocabulary.restore_text(self.output)
                self.update_label(inp, res)
                self.display_text.config(state=tk.NORMAL)
                self.display_text.insert(tk.END, self.text_history[-2] + self.text_history[-1])
                self.display_text.config(state=tk.DISABLED)
                self.display_text.see(tk.END)
        else:
            self.entry.set('')
            if self.auto_inp == None and inp != '':
                self.auto_inp = inp
                self.main_button.config(text='Next')

            self.output, self.input, attention_plot = evaluate(self.auto_inp, self.v, self.enc, self.dec, self.hparams['MAX_LEN'])
            self.attention_weights = attention_plot[:len(self.output.split(' ')), :len(self.input.split(' '))]

            res = Vocabulary.restore_text(self.output)
            self.update_label(self.auto_inp, res)
            self.display_text.config(state=tk.NORMAL)
            self.display_text.insert(tk.END, self.text_history[-2] + self.text_history[-1])
            self.display_text.config(state=tk.DISABLED)
            self.display_text.see(tk.END)

            self.auto_inp = res
  
    def update_label(self, inp, result):
        if self.mode == 'chat':
            self.text_history.append('-> YOU: ' + inp + '\n')
            self.text_history.append('>> BOT: ' + result + '\n\n')
        else:
            self.text_history.append('-> INP: ' + inp + '\n')
            self.text_history.append('>> REZ: ' + result + '\n\n')

    def attention_button_fnc(self):
        if self.input != None and self.output != None:
            plot_attention(self.attention_weights, self.input.split(' '), self.output.split(' '))

    def auto_button_fnc(self):
        self.reset_states()

        if self.mode == 'chat':
            self.main_button.config(text='Start')
            self.auto_button.config(text='Normal Chat')
            self.label_text.set('       Auto mode')
            self.mode = 'auto'
            self.reset_button.pack(side=tk.RIGHT)
        else:
            self.main_button.config(text='Send')
            self.auto_button.config(text='Auto Chat')
            self.label_text.set('Chat mode')
            self.mode = 'chat'
            self.auto_inp = None
            self.reset_button.pack_forget()

    def reset_button_fnc(self):
        self.reset_states()
        self.main_button.config(text='Start')
        self.auto_inp = None

    def reset_states(self):
        self.display_text.config(state=tk.NORMAL)
        self.display_text.delete(1.0, tk.END)
        self.display_text.config(state=tk.DISABLED)
        self.text_history = []

    def start(self):
        self.load_model()
        self.reset_states()
        self.entry_w.config(state=tk.NORMAL)
        self.label_text.set('Chat mode')
        self.root.mainloop()


if __name__ == '__main__':
    if os.path.isdir('./__pycache__/'): shutil.rmtree(path='./__pycache__/', ignore_errors=True, onerror=None)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    AppWindow().start()
