from kivy.lang import Builder

from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.app import MDApp

#name window and connect the Root MDApp to the .kv file home page
class AutoPTZ(MDApp):
    def build(self):
        self.theme_cls.theme_style = "Dark"
        return Builder.load_file("test_homepage.kv")


AutoPTZ().run()