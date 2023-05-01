from .frame import Frame

class MisinfoReactionFrame(Frame):
    def __init__(self, content, intent, reaction, misinfo):
        super().__init__(content, intent, reaction)
        self.misinfo = misinfo

    def __str__(self):
        return f"Content: {self.content}\nIntent: {self.intent}\nReaction: {self.reaction}\nMisinfo: {self.misinfo}"

    def __repr__(self):
        return f"Content: {self.content}\nIntent: {self.intent}\nReaction: {self.reaction}\nMisinfo: {self.misinfo}"