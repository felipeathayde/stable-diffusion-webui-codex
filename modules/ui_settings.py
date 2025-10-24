from __future__ import annotations
import gradio as gr
from typing import Any, Callable, Iterable, Optional

import html

from modules import ui_common, shared, script_callbacks, scripts, sd_models, sysinfo, timer, shared_items
from modules.call_queue import wrap_gradio_call_no_job
from modules.options import options_section
from modules.shared import opts
from modules.ui_components import FormRow
from modules.ui_gradio_extensions import reload_javascript
from concurrent.futures import ThreadPoolExecutor, as_completed
from modules_forge import main_entry


def get_value_for_setting(key: str) -> gr.Update:
    value = getattr(opts, key)

    info = opts.data_labels[key]
    args = info.component_args() if callable(info.component_args) else info.component_args or {}
    args = {k: v for k, v in args.items() if k not in {'precision'}}

    return gr.update(value=value, **args)


def _normalize_choices(raw_choices: Iterable[Any]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for item in raw_choices:
        if isinstance(item, dict):
            value = item.get('value', '')
            label = item.get('label', value)
        elif isinstance(item, (list, tuple)) and item:
            value = item[0]
            label = item[1] if len(item) > 1 else item[0]
        else:
            value = item
            label = item
        normalized.append({
            "value": str(value),
            "label": str(label),
        })
    return normalized


def _render_native_dropdown_html(*, key: str, label: str | None, elem_id: str, choices: Iterable[Any], value: Any, allow_custom: bool) -> str:
    normalized = _normalize_choices(list(choices))
    value_str = '' if value is None else str(value)
    options_html = []
    for choice in normalized:
        opt_value = html.escape(choice['value'], quote=True)
        opt_label = html.escape(choice['label'], quote=True)
        label_attr = f' label="{opt_label}"' if opt_label and opt_label != opt_value else ''
        options_html.append(f'<option value="{opt_value}"{label_attr}></option>')

    allow_attr = "1" if allow_custom else "0"
    label_text = html.escape(label or key)
    input_id = f"{elem_id}__input"
    list_id = f"{elem_id}__list"

    return (
        f'<div class="sdw-native-dropdown" data-target="{html.escape(elem_id)}" '
        f'data-key="{html.escape(key)}" data-allow-custom="{allow_attr}">'  # type: ignore[arg-type]
        f'<label class="sdw-native-dropdown__label" for="{input_id}">{label_text}</label>'
        f'<div class="sdw-native-dropdown__control">'
        f'<input id="{input_id}" class="sdw-native-dropdown__input" '
        f'list="{list_id}" autocomplete="off" value="{html.escape(value_str, quote=True)}" />'
        f'<button type="button" class="sdw-native-dropdown__toggle" aria-label="Toggle options" tabindex="-1"></button>'
        f'</div>'
        f'<datalist id="{list_id}">{"".join(options_html)}</datalist>'
        f'</div>'
    )


def create_setting_component(key: str, is_quicksettings: bool = False):
    def fun() -> Any:
        return opts.data[key] if key in opts.data else opts.data_labels[key].default

    info = opts.data_labels[key]
    t = type(info.default)

    raw_args = info.component_args() if callable(info.component_args) else info.component_args
    args = dict(raw_args) if isinstance(raw_args, dict) else (dict(raw_args) if raw_args else {})

    elem_classes = list(args.get('elem_classes', []))

    choices_for_dropdown: Iterable[Any] | None = None
    allow_custom = bool(args.get('allow_custom_value', False))
    multiselect = bool(args.get('multiselect', False))

    if info.component is not None:
        comp = info.component
    elif t == str:
        comp = gr.Textbox
    elif t == int:
        comp = gr.Number
    elif t == bool:
        comp = gr.Checkbox
    else:
        raise Exception(f'bad options item type: {t} for key {key}')

    use_native_dropdown = False

    if not multiselect:
        if comp in {gr.Textbox, gr.Number}:
            choices_for_dropdown = args.get("choices")
            if choices_for_dropdown is not None:
                comp = gr.Textbox
                use_native_dropdown = True
        else:
            try:
                if issubclass(comp, gr.Dropdown):
                    choices_for_dropdown = args.get("choices")
                    use_native_dropdown = True
                    comp = gr.Textbox
            except TypeError:
                pass

    if use_native_dropdown:
        # remove gradio dropdown specific args
        args.pop('choices', None)
        args.pop('allow_custom_value', None)
        args.pop('multiselect', None)
        allow_custom = bool(allow_custom)
        elem_classes = [c for c in elem_classes if c not in {'sdw-native-select-target', 'sdw-native-select-allow-custom'}]

    if elem_classes:
        args['elem_classes'] = elem_classes

    elem_id = f"setting_{key}"

    if comp == gr.State:
        return gr.State(fun())

    dropdown_html_component = None
    dropdown_choices = _normalize_choices(choices_for_dropdown or []) if choices_for_dropdown is not None else []

    if use_native_dropdown and dropdown_choices:
        value = fun()
        dropdown_html_component = gr.HTML(
            _render_native_dropdown_html(
                key=key,
                label=info.label,
                elem_id=elem_id,
                choices=dropdown_choices,
                value=value,
                allow_custom=allow_custom,
            ),
            elem_id=f"{elem_id}__dropdown",
            elem_classes=['sdw-native-dropdown-host'],
            render=False,
            container=True,
        )
        if is_quicksettings:
            dropdown_html_component.render()
        args = dict(args)
        hidden_classes = list(args.get('elem_classes', []))
        if 'sdw-native-dropdown-value' not in hidden_classes:
            hidden_classes.append('sdw-native-dropdown-value')
        args['elem_classes'] = hidden_classes
        args.setdefault('visible', False)
        args.setdefault('label', info.label)
        args.setdefault('value', value)
        res = gr.Textbox(**args)
        res.native_dropdown_choices = dropdown_choices
        res.native_dropdown_allow_custom = allow_custom
        res.native_dropdown_key = key
        res.native_dropdown_label = info.label
        res.native_dropdown_html = dropdown_html_component
        res.native_dropdown_elem_id = elem_id
        res.native_dropdown_value = value
    else:
        component_kwargs = dict(args)
        component_kwargs.setdefault('label', info.label)
        component_kwargs.setdefault('value', fun())
        component_kwargs.setdefault('elem_id', elem_id)
        res = comp(**component_kwargs)

    if info.refresh is not None:
        refresh_components: list[gr.components.Component] = [res]
        html_component = getattr(res, 'native_dropdown_html', None)
        if html_component is not None:
            refresh_components.append(html_component)

        def _refreshed_args():
            refreshed = info.component_args() if callable(info.component_args) else info.component_args or {}
            if isinstance(refreshed, dict):
                refreshed_args = dict(refreshed)
            else:
                refreshed_args = {k: v for k, v in getattr(refreshed, "__dict__", {}).items() if not k.startswith('_')}

            new_value = refreshed_args.get('value', getattr(opts, key))
            dropdown_update = {'value': new_value}
            if 'visible' in refreshed_args:
                dropdown_update['visible'] = refreshed_args['visible']
            if 'interactive' in refreshed_args:
                dropdown_update['interactive'] = refreshed_args['interactive']

            html_update = None
            if html_component is not None:
                choices_raw = refreshed_args.get('choices', res.native_dropdown_choices)
                choices_norm = _normalize_choices(choices_raw) if choices_raw is not None else res.native_dropdown_choices
                allow = bool(refreshed_args.get('allow_custom_value', res.native_dropdown_allow_custom))
                html_update = _render_native_dropdown_html(
                    key=key,
                    label=info.label,
                    elem_id=elem_id,
                    choices=choices_norm,
                    value=new_value,
                    allow_custom=allow,
                )
                res.native_dropdown_choices = choices_norm
                res.native_dropdown_allow_custom = allow
                res.native_dropdown_value = new_value

            dropdown_update.pop('choices', None)
            dropdown_update.pop('allow_custom_value', None)
            dropdown_update.pop('multiselect', None)

            updates: list[Any]
            if html_update is not None:
                updates = [dropdown_update, {'value': html_update}]
            else:
                updates = [dropdown_update]
            return updates

        if is_quicksettings:
            ui_common.create_refresh_button(refresh_components, info.refresh, _refreshed_args, f"refresh_{key}")
        else:
            with FormRow():
                html_component and html_component.render()
                ui_common.create_refresh_button(refresh_components, info.refresh, _refreshed_args, f"refresh_{key}")
    else:
        if dropdown_html_component is not None and not is_quicksettings:
            dropdown_html_component.render()

    return res


class UiSettings:
    submit = None
    result = None
    interface = None
    components = None
    component_dict = None
    dummy_component = None
    quicksettings_list = None
    quicksettings_names = None
    text_settings = None
    show_all_pages = None
    show_one_page = None
    search_input = None

    def run_settings(self, *args: Any):
        changed: list[str] = []

        for key, value, comp in zip(opts.data_labels.keys(), args, self.components):
            assert comp == self.dummy_component or opts.same_type(value, opts.data_labels[key].default), f"Bad value for setting {key}: {value}; expecting {type(opts.data_labels[key].default).__name__}"

        for key, value, comp in zip(opts.data_labels.keys(), args, self.components):
            if comp == self.dummy_component:
                continue

            # don't set (Managed by Forge) options, they revert to defaults
            if key in ["sd_model_checkpoint", "CLIP_stop_at_last_layers", "sd_vae"]:
                continue

            if opts.set(key, value):
                changed.append(key)

        try:
            opts.save(shared.config_filename)
        except RuntimeError:
            return opts.dumpjson(), f'{len(changed)} settings changed without save: {", ".join(changed)}.'
        return opts.dumpjson(), f'{len(changed)} settings changed{": " if changed else ""}{", ".join(changed)}.'

    def run_settings_single(self, value: Any, key: str):
        if not opts.same_type(value, opts.data_labels[key].default):
            return gr.update(visible=True), opts.dumpjson()

        if value is None or not opts.set(key, value):
            return gr.update(value=getattr(opts, key)), opts.dumpjson()

        opts.save(shared.config_filename)

        return get_value_for_setting(key), opts.dumpjson()

    def register_settings(self) -> None:
        script_callbacks.ui_settings_callback()

    def create_ui(self, loadsave: Any, dummy_component: gr.components.Component):
        self.components = []
        self.component_dict = {}
        self.dummy_component = dummy_component

        shared.settings_components = self.component_dict

        # we add this as late as possible so that scripts have already registered their callbacks
        opts.data_labels.update(options_section(('callbacks', "Callbacks", "system"), {
            **shared_items.callbacks_order_settings(),
        }))

        opts.reorder()

        with gr.Blocks(analytics_enabled=False) as settings_interface:
            with gr.Row():
                with gr.Column(scale=6):
                    self.submit = gr.Button(value="Apply settings", variant='primary', elem_id="settings_submit")
                with gr.Column():
                    restart_gradio = gr.Button(value='Reload UI', variant='primary', elem_id="settings_restart_gradio")

            self.result = gr.HTML(elem_id="settings_result")

            self.quicksettings_names = opts.quick_setting_list
            self.quicksettings_names = {x: i for i, x in enumerate(self.quicksettings_names) if x != 'quicksettings'}

            self.quicksettings_list = []

            previous_section = None
            current_tab = None
            current_row = None
            pair_open = False
            saving_sections = []  # (elem_id, text)
            with gr.Tabs(elem_id="settings"):
                for i, (k, item) in enumerate(opts.data_labels.items()):
                    section_must_be_skipped = item.section[0] is None

                    if item.category_id == 'saving':
                        # Defer rendering of 'saving' sections into a grouped tab later
                        sec_id, sec_text = item.section
                        if (sec_id, sec_text) not in saving_sections:
                            saving_sections.append((sec_id, sec_text))
                        self.components.append(dummy_component)
                        continue

                    if previous_section != item.section and not section_must_be_skipped:
                        elem_id, text = item.section

                        if current_tab is not None:
                            current_row.__exit__()
                            current_tab.__exit__()

                        gr.Group()
                        current_tab = gr.TabItem(elem_id=f"settings_{elem_id}", label=text)
                        current_tab.__enter__()
                        current_row = gr.Column(elem_id=f"column_settings_{elem_id}", variant='compact')
                        current_row.__enter__()

                        previous_section = item.section

                    if k in self.quicksettings_names and not shared.cmd_opts.freeze_settings:
                        self.quicksettings_list.append((i, k, item))
                        self.components.append(dummy_component)
                    elif section_must_be_skipped:
                        self.components.append(dummy_component)
                    else:
                        # Pair simple components two per row for denser layout.
                        # Do not pair if the option requires a refresh button (handled inside create_setting_component).
                        info = opts.data_labels[k]
                        if getattr(info, 'refresh', None) is None:
                            if not pair_open:
                                pair_open = True
                                row_ctx = gr.Row(variant='compact', elem_id=f"row_settings_{elem_id}_{i}")
                                row_ctx.__enter__()
                                # first column
                                with gr.Column(scale=1):
                                    component = create_setting_component(k)
                            else:
                                # second column, close row after
                                with gr.Column(scale=1):
                                    component = create_setting_component(k)
                                row_ctx.__exit__()
                                pair_open = False
                        else:
                            # non-pairable (has refresh or custom), ensure any open pair is closed
                            if pair_open:
                                row_ctx.__exit__()
                                pair_open = False
                            component = create_setting_component(k)

                        self.component_dict[k] = component
                        self.components.append(component)

                # Close any dangling row
                if pair_open:
                    try:
                        row_ctx.__exit__()
                    except Exception:
                        pass
                    pair_open = False

                if current_tab is not None:
                    current_row.__exit__()
                    current_tab.__exit__()

                # Render grouped 'Saving' tab with accordions, if any
                if saving_sections:
                    with gr.TabItem(elem_id="settings_cat_saving", label="Saving"):
                        for sec_id, sec_text in saving_sections:
                            with gr.Accordion(sec_text, open=False, elem_id=f"acc_{sec_id}"):
                                with gr.Column(elem_id=f"column_settings_{sec_id}", variant='compact'):
                                    # Render all options belonging to this section
                                    # Iterate labels in order and materialize components
                                    for k2, item2 in opts.data_labels.items():
                                        if item2.section == (sec_id, sec_text) and item2.category_id == 'saving':
                                            info2 = item2
                                            if getattr(info2, 'refresh', None) is None:
                                                with gr.Row(variant='compact'):
                                                    with gr.Column(scale=1):
                                                        comp = create_setting_component(k2)
                                                    self.component_dict[k2] = comp
                                                    self.components.append(comp)
                                            else:
                                                comp = create_setting_component(k2)
                                                self.component_dict[k2] = comp
                                                self.components.append(comp)

                with gr.TabItem("Defaults", id="defaults", elem_id="settings_tab_defaults"):
                    loadsave.create_ui()

                with gr.TabItem("Sysinfo", id="sysinfo", elem_id="settings_tab_sysinfo"):
                    gr.HTML('<a href="./internal/sysinfo-download" class="sysinfo_big_link" download>Download system info</a><br /><a href="./internal/sysinfo" target="_blank">(or open as text in a new page)</a>', elem_id="sysinfo_download")

                    with gr.Row():
                        with gr.Column(scale=1):
                            sysinfo_check_file = gr.File(label="Check system info for validity", type='binary')
                        with gr.Column(scale=1):
                            sysinfo_check_output = gr.HTML("", elem_id="sysinfo_validity")
                        with gr.Column(scale=100):
                            pass

                    # Live memory/VRAM diagnostics
                    with gr.Row():
                        mem_refresh = gr.Button(value='Refresh memory', elem_id="sysinfo_refresh_memory")
                    with gr.Row():
                        mem_json = gr.JSON(value={}, elem_id="sysinfo_memory_json")

                with gr.TabItem("Actions", id="actions", elem_id="settings_tab_actions"):
                    request_notifications = gr.Button(value='Request browser notifications', elem_id="request_notifications")
                    download_localization = gr.Button(value='Download localization template', elem_id="download_localization")
                    reload_script_bodies = gr.Button(value='Reload custom script bodies (No ui updates, No restart)', variant='secondary', elem_id="settings_reload_script_bodies")
                    with gr.Row():
                        unload_sd_model = gr.Button(value='Unload all models', elem_id="sett_unload_sd_model")
#                        reload_sd_model = gr.Button(value='Load SD checkpoint to VRAM from RAM', elem_id="sett_reload_sd_model")
                    with gr.Row():
                        calculate_all_checkpoint_hash = gr.Button(value='Calculate hash for all checkpoint', elem_id="calculate_all_checkpoint_hash")
                        calculate_all_checkpoint_hash_threads = gr.Number(value=1, label="Number of parallel calculations", elem_id="calculate_all_checkpoint_hash_threads", precision=0, minimum=1)

                with gr.TabItem("Licenses", id="licenses", elem_id="settings_tab_licenses"):
                    gr.HTML(shared.html("licenses.html"), elem_id="licenses")

                self.show_all_pages = gr.Button(value="Show all pages", elem_id="settings_show_all_pages")
                self.show_one_page = gr.Button(value="Show only one page", elem_id="settings_show_one_page", visible=False)
                self.show_one_page.click(lambda: None)

                self.search_input = gr.Textbox(value="", elem_id="settings_search", max_lines=1, placeholder="Search...", show_label=False)

                self.text_settings = gr.Textbox(elem_id="settings_json", value=lambda: opts.dumpjson(), visible=False)

            def call_func_and_return_text(func: Callable[[], None], text: str):
                def handler() -> str:
                    t = timer.Timer()
                    func()
                    t.record(text)

                    return f'{text} in {t.total:.1f}s'

                return handler

            unload_sd_model.click(
                fn=call_func_and_return_text(sd_models.unload_model_weights, 'Unloaded all models'),
                inputs=[],
                outputs=[self.result]
            )

#            reload_sd_model.click(
#                fn=call_func_and_return_text(lambda: sd_models.send_model_to_device(shared.sd_model), 'Loaded the checkpoint'),
#                inputs=[],
#                outputs=[self.result]
#            )

            request_notifications.click(
                fn=lambda: None,
                inputs=[],
                outputs=[],
                _js='function(){}'
            )

            download_localization.click(
                fn=lambda: None,
                inputs=[],
                outputs=[],
                _js='download_localization'
            )

            def reload_scripts():
                scripts.reload_script_body_only()
                reload_javascript()  # need to refresh the html page

            reload_script_bodies.click(
                fn=reload_scripts,
                inputs=[],
                outputs=[]
            )

            restart_gradio.click(
                fn=shared.state.request_restart,
                _js='restart_reload',
                inputs=[],
                outputs=[],
            )

            def check_file(x: Optional[bytes]) -> str:
                if x is None:
                    return ''

                if sysinfo.check(x.decode('utf8', errors='ignore')):
                    return 'Valid'

                return 'Invalid'

            sysinfo_check_file.change(
                fn=check_file,
                inputs=[sysinfo_check_file],
                outputs=[sysinfo_check_output],
            )

            def get_internal_memory() -> dict[str, Any]:
                try:
                    import os as _os
                    import psutil as _ps
                    _proc = _ps.Process(_os.getpid())
                    _rss = _proc.memory_info().rss
                    _pct = _proc.memory_percent()
                    _ram_total = int(100 * _rss / _pct) if _pct > 0 else _ps.virtual_memory().total
                    _ram = {"free": _ram_total - _rss, "used": _rss, "total": _ram_total}
                except Exception as _e:
                    _ram = {"error": str(_e)}

                try:
                    import torch as _torch
                    if _torch.cuda.is_available():
                        free_b, total_b = _torch.cuda.mem_get_info()
                        _cuda = {"system": {"free": free_b, "used": total_b - free_b, "total": total_b}}
                    else:
                        _cuda = {"error": "unavailable"}
                except Exception as _e:
                    _cuda = {"error": str(_e)}

                try:
                    from apps.server.backend import memory_management as _mm
                    vram_state = getattr(_mm, 'vram_state', None)
                    cpu_state = getattr(_mm, 'cpu_state', None)
                    flags = {
                        "vram_state": getattr(vram_state, 'name', str(vram_state)),
                        "cpu_state": getattr(cpu_state, 'name', str(cpu_state)),
                        "pin_shared_memory": getattr(_mm, 'PIN_SHARED_MEMORY', None),
                        "always_offload_from_vram": getattr(_mm, 'ALWAYS_VRAM_OFFLOAD', None),
                    }
                except Exception:
                    flags = {}

                return {"ram": _ram, "cuda": _cuda, "flags": flags}

            mem_refresh.click(fn=get_internal_memory, inputs=[], outputs=[mem_json], show_progress=False, queue=False)

            # Native settings search (Python-side) — avoids JS DOM filtering.
            # Returns visibility updates for each settings component.
            if self.components:
                self.search_input.change(
                    fn=self.search,
                    inputs=[self.search_input],
                    outputs=self.components,
                    show_progress=False,
                    queue=False,
                )

            def calculate_all_checkpoint_hash_fn(max_thread: int) -> None:
                checkpoints_list = sd_models.checkpoints_list.values()
                with ThreadPoolExecutor(max_workers=max_thread) as executor:
                    futures = [executor.submit(checkpoint.calculate_shorthash) for checkpoint in checkpoints_list]
                    completed = 0
                    for _ in as_completed(futures):
                        completed += 1
                        print(f"{completed} / {len(checkpoints_list)} ")
                    print("Finish calculating hash for all checkpoints")

            calculate_all_checkpoint_hash.click(
                fn=calculate_all_checkpoint_hash_fn,
                inputs=[calculate_all_checkpoint_hash_threads],
            )

        self.interface = settings_interface

    def add_quicksettings(self):
        with gr.Row(elem_id="quicksettings", variant="compact") as quicksettings_row:
            # Reintroduce checkpoint/vae selectors as in master
            from modules_forge import main_entry
            main_entry.make_checkpoint_manager_ui()

            managed_keys = {"forge_selected_vae", "forge_additional_modules"}
            for _i, k, _item in sorted(self.quicksettings_list, key=lambda x: self.quicksettings_names.get(x[1], x[0])):
                if k in managed_keys:
                    continue
                component = create_setting_component(k, is_quicksettings=True)
                self.component_dict[k] = component
        return quicksettings_row

    def add_functionality(self, demo: gr.Blocks) -> None:
        # In Settings v2, the legacy settings interface may not be mounted.
        # Only wire the legacy submit button if it exists in the active Blocks.
        if getattr(self, 'submit', None) is not None and hasattr(self.submit, 'click'):
            self.submit.click(
                fn=wrap_gradio_call_no_job(lambda *args: self.run_settings(*args), extra_outputs=[gr.update()]),
                inputs=self.components,
                outputs=[self.text_settings, self.result],
            )

        for _i, k, _item in self.quicksettings_list:
            component = self.component_dict[k]

            if hasattr(component, 'native_dropdown_html'):
                methods = [component.change]
            elif isinstance(component, gr.Textbox):
                methods = [component.submit, component.blur]
            elif hasattr(component, 'release'):
                methods = [component.release]
            else:
                methods = [component.change]

            for method in methods:
                method(
                    fn=lambda value, k=k: self.run_settings_single(value, key=k),
                    inputs=[component],
                    outputs=[component, self.text_settings],
                    show_progress=False,
                )

        # Wire Change checkpoint button to minimal checkpoint/vae selectors if present
        try:
            from modules_forge import main_entry

            def button_set_checkpoint_change(model, vae, text_modules, dummy):
                model = sd_models.match_checkpoint_to_name(model)
                if isinstance(vae, list):
                    vae = vae[0] if vae else 'Automatic'
                if vae == 'Built in':
                    vae = 'Automatic'
                if isinstance(text_modules, str):
                    text_modules = [text_modules]
                return model, vae, text_modules or [], opts.dumpjson()

            button_set_checkpoint = gr.Button('Change checkpoint', elem_id='change_checkpoint', visible=False)
            button_set_checkpoint.click(
                fn=button_set_checkpoint_change,
                js="function(c, v, te, n){ var ckpt = (desiredCheckpointName !== null && desiredCheckpointName !== undefined) ? desiredCheckpointName : c; var vae = (desiredVAEName !== null && desiredVAEName !== undefined) ? desiredVAEName : v; var modules = Array.isArray(desiredVAEExtras) ? desiredVAEExtras : te; desiredCheckpointName = null; desiredVAEName = null; desiredVAEExtras = null; return [ckpt, vae, modules, null]; }",
                inputs=[main_entry.ui_checkpoint, main_entry.ui_vae, main_entry.ui_text_encoders, self.dummy_component],
                outputs=[main_entry.ui_checkpoint, main_entry.ui_vae, main_entry.ui_text_encoders, self.text_settings],
            )
        except Exception:
            pass

        component_keys = [k for k in opts.data_labels.keys() if k in self.component_dict]

        def get_settings_values():
            return [get_value_for_setting(key) for key in component_keys]

        demo.load(
            fn=get_settings_values,
            inputs=[],
            outputs=[self.component_dict[k] for k in component_keys],
            queue=False,
        )

    def search(self, text: str):
        print(text)

        return [gr.update(visible=text in (comp.label or "")) for comp in self.components]
