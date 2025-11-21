from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from tkinter import END, Button, Entry, Frame, Label, Listbox, Menu, SINGLE, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError as exc:
    raise ImportError(
        "Dependência ausente: 'tkinterdnd2'. "
        "Instale com 'pip install tkinterdnd2'."
    ) from exc

from PIL import Image, PngImagePlugin


class MetadataEditorApp(TkinterDnD.Tk):
    def __init__(self) -> None:
        super().__init__()

        self.title("Editor de metadata PNG (Stable Diffusion)")
        self.geometry("900x600")

        self.image_path: Optional[Path] = None
        self.image: Optional[Image.Image] = None
        self.text_metadata: Dict[str, str] = {}
        self.current_key: Optional[str] = None

        self._build_menu()
        self._build_widgets()

    def _build_menu(self) -> None:
        menubar = Menu(self)
        file_menu = Menu(menubar, tearoff=0)
        file_menu.add_command(label="Abrir...", command=self.open_file_dialog)
        file_menu.add_command(label="Salvar", command=self.save_image)
        file_menu.add_command(label="Salvar como...", command=self.save_image_as)
        file_menu.add_separator()
        file_menu.add_command(label="Sair", command=self.quit)
        menubar.add_cascade(label="Arquivo", menu=file_menu)
        self.config(menu=menubar)

    def _build_widgets(self) -> None:
        top = Frame(self)
        top.pack(fill="x", padx=8, pady=8)

        self.drop_label = Label(
            top,
            text="Arraste um arquivo PNG aqui ou use Arquivo → Abrir...",
            relief="groove",
            padx=8,
            pady=8,
        )
        self.drop_label.pack(side="left", fill="both", expand=True)

        open_button = Button(
            top,
            text="Abrir PNG...",
            command=self.open_file_dialog,
        )
        open_button.pack(side="right", padx=(8, 0))

        # Permitir arrastar tanto sobre o rótulo quanto sobre a janela inteira.
        self.drop_target_register(DND_FILES)
        self.dnd_bind("<<Drop>>", self.on_drop)
        self.drop_label.drop_target_register(DND_FILES)
        self.drop_label.dnd_bind("<<Drop>>", self.on_drop)

        main = Frame(self)
        main.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        left = Frame(main)
        left.pack(side="left", fill="y")

        Label(left, text="Chaves de metadata:").pack(anchor="w")

        self.key_listbox = Listbox(left, selectmode=SINGLE, width=30)
        self.key_listbox.pack(fill="y", expand=False)
        self.key_listbox.bind("<<ListboxSelect>>", self.on_key_selected)

        buttons = Frame(left)
        buttons.pack(fill="x", pady=(8, 0))

        Button(buttons, text="Nova chave", command=self.add_key).pack(
            side="left", expand=True, fill="x"
        )
        Button(buttons, text="Remover", command=self.remove_key).pack(
            side="left", expand=True, fill="x"
        )

        right = Frame(main)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))

        Label(right, text="Valor da metadata:").pack(anchor="w")

        search_frame = Frame(right)
        search_frame.pack(fill="x", pady=(0, 4))

        Label(search_frame, text="Localizar:").pack(side="left")

        self.search_entry = Entry(search_frame)
        self.search_entry.pack(side="left", fill="x", expand=True, padx=(4, 4))
        self.search_entry.bind("<Return>", self.highlight_search)

        Button(
            search_frame,
            text="Destacar",
            command=self.highlight_search,
        ).pack(side="left")

        self.value_text = ScrolledText(right, wrap="word")
        self.value_text.pack(fill="both", expand=True)
        self.value_text.tag_config("search_highlight", background="yellow", foreground="black")

        Button(
            right,
            text="Aplicar alterações nesta chave",
            command=self.update_current_key_value,
        ).pack(anchor="e", pady=(8, 0))

    def open_file_dialog(self) -> None:
        path = filedialog.askopenfilename(
            title="Escolher imagem PNG",
            filetypes=[("PNG", "*.png"), ("Todos os arquivos", "*.*")],
        )
        if path:
            self.load_image(Path(path))

    def on_drop(self, event) -> None:
        # event.data pode conter vários ficheiros
        paths = self.splitlist(event.data)
        if not paths:
            return
        first = paths[0]
        self.load_image(Path(first))

    def load_image(self, path: Path) -> None:
        if not path.exists():
            messagebox.showerror("Erro", f"Arquivo não encontrado: {path}")
            return

        try:
            img = Image.open(path)
            img.load()
        except Exception as exc:
            messagebox.showerror("Erro ao abrir imagem", str(exc))
            return

        text_metadata: Dict[str, str] = {}
        for key, value in img.info.items():
            if isinstance(value, str):
                text_metadata[key] = value

        if not text_metadata:
            # Imagens do WebUI normalmente possuem o campo "parameters".
            text_metadata["parameters"] = ""

        self.image_path = path
        self.image = img
        self.text_metadata = text_metadata
        self.current_key = None

        self.refresh_metadata_ui()
        self.title(f"Editor de metadata PNG – {path.name}")

    def refresh_metadata_ui(self) -> None:
        self.key_listbox.delete(0, END)
        for key in sorted(self.text_metadata.keys()):
            self.key_listbox.insert(END, key)

        self.value_text.delete("1.0", END)
        self.value_text.tag_remove("search_highlight", "1.0", END)
        if self.text_metadata:
            self.key_listbox.selection_set(0)
            self.on_key_selected(None)

    def on_key_selected(self, event) -> None:
        selection = self.key_listbox.curselection()
        if not selection:
            return

        index = selection[0]
        key = self.key_listbox.get(index)
        self.current_key = key

        value = self.text_metadata.get(key, "")
        self.value_text.delete("1.0", END)
        self.value_text.insert("1.0", value)
        self.value_text.tag_remove("search_highlight", "1.0", END)

    def highlight_search(self, event=None) -> None:
        query = self.search_entry.get().strip() if hasattr(self, "search_entry") else ""
        self.value_text.tag_remove("search_highlight", "1.0", END)
        if not query:
            return

        start = "1.0"
        first_index = None
        while True:
            idx = self.value_text.search(query, start, stopindex=END, nocase=1)
            if not idx:
                break
            end = f"{idx}+{len(query)}c"
            self.value_text.tag_add("search_highlight", idx, end)
            if first_index is None:
                first_index = idx
            start = end

        if first_index is not None:
            self.value_text.see(first_index)
            self.value_text.mark_set("insert", first_index)

    def _apply_pending_edits(self) -> None:
        """
        Copia o conteúdo atual do campo de texto
        para o dicionário de metadata da chave selecionada.
        """
        if not self.current_key:
            return

        new_value = self.value_text.get("1.0", END).rstrip("\n")
        self.text_metadata[self.current_key] = new_value

    def update_current_key_value(self) -> None:
        if not self.current_key:
            messagebox.showwarning(
                "Nenhuma chave selecionada",
                "Selecione uma chave de metadata antes de aplicar alterações.",
            )
            return

        self._apply_pending_edits()
        # Após aplicar as alterações da chave selecionada,
        # salva imediatamente a imagem no arquivo atual.
        # Isso deixa o fluxo mais simples: editar → aplicar → arquivo persistido.
        self.save_image()

    def add_key(self) -> None:
        from tkinter import simpledialog

        # Garante que a chave atual não perca edições ao criar outra.
        self._apply_pending_edits()

        new_key = simpledialog.askstring("Nova chave", "Nome da nova chave:")
        if not new_key:
            return

        if new_key in self.text_metadata:
            messagebox.showerror(
                "Chave existente",
                f"Já existe uma chave chamada '{new_key}'.",
            )
            return

        self.text_metadata[new_key] = ""
        self.refresh_metadata_ui()

    def remove_key(self) -> None:
        self._apply_pending_edits()

        if not self.current_key:
            messagebox.showwarning(
                "Nenhuma chave selecionada",
                "Selecione uma chave para remover.",
            )
            return

        confirm = messagebox.askyesno(
            "Remover chave",
            f"Remover a chave '{self.current_key}' da metadata?",
        )
        if not confirm:
            return

        self.text_metadata.pop(self.current_key, None)
        self.current_key = None
        self.refresh_metadata_ui()

    def _build_pnginfo(self) -> PngImagePlugin.PngInfo:
        pnginfo = PngImagePlugin.PngInfo()
        for key, value in self.text_metadata.items():
            pnginfo.add_text(key, value)
        return pnginfo

    def save_image(self) -> None:
        if self.image is None or self.image_path is None:
            messagebox.showwarning(
                "Nenhuma imagem",
                "Carregue uma imagem antes de salvar.",
            )
            return

        # Garante que o conteúdo editado na caixa de texto
        # esteja sincronizado com o dicionário antes de salvar.
        self._apply_pending_edits()
        self._save_to_path(self.image_path)

    def save_image_as(self) -> None:
        if self.image is None:
            messagebox.showwarning(
                "Nenhuma imagem",
                "Carregue uma imagem antes de salvar.",
            )
            return

        self._apply_pending_edits()

        path_str = filedialog.asksaveasfilename(
            title="Salvar imagem PNG",
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
            initialfile=self.image_path.name if self.image_path else "output.png",
        )
        if not path_str:
            return
        self._save_to_path(Path(path_str))

    def _save_to_path(self, path: Path) -> None:
        if self.image is None:
            return

        pnginfo = self._build_pnginfo()

        try:
            self.image.save(path, pnginfo=pnginfo)
        except Exception as exc:
            messagebox.showerror("Erro ao salvar imagem", str(exc))
            return

        # Log simples no console para facilitar debug.
        print(f"[png-metadata-editor] imagem salva em: {path}")

        self.image_path = path
        self.title(f"Editor de metadata PNG – {path.name}")
        messagebox.showinfo("Salvo", f"Imagem salva em\n{path}")


def main() -> None:
    app = MetadataEditorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
