# magentic-two
A local take on autogen magentic-one with custom ollama connection using llama3.2-vision

playwright install --with-deps chromium
### Notes
1. activate environment: source ../../.venv/bin/activate
2. Start docker desktop
3. Use the create local agent method from utils.py
4. Use Ollama chat client with vision support.
5. Use forcing locale of playwright chromium:

### task
I want you to pick 5 stocks that you think will perform well in the next 6 weeks. Please provide a brief explanation for each stock. I want you to go over reddit.com/r/wallstreetbets and the website stockwits along with twitter to find the most discussed stocks. The stocks need to be cheap still, that is, close to the value at IPO, have not been pump and dump yet, and the companies are not bankrupt or shady. you need to pick stocks for holding positions only, no shorts, no options. Good examples for stocks that worked well recently are: RKLB, ASTS, LUNR, MTEK, SATL. Only write and use code if you find it necessary, most of the tasks can be done by reading and summarizing information.
