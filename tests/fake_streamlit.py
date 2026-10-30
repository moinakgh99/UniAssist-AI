"""A minimal fake of the `streamlit` module, enough to execute src/app.py + ui_components.py in a test.

It models: session_state, scripted button clicks (with on_click callbacks), widget state by key, st.rerun(),
st.stop(), caching decorators, layout contexts, and it raises on DUPLICATE widget keys just like Streamlit.
It does NOT render anything - it checks control flow, backend calls and the HTML strings that would be sent.
"""
import runpy
import sys
import types


class RerunException(Exception):
    pass


class StopScript(Exception):
    pass


class DuplicateWidgetID(Exception):
    pass


class SessionState(dict):
    __getattr__ = lambda self, k: self[k] if k in self else (_ for _ in ()).throw(AttributeError(k))
    __setattr__ = dict.__setitem__

    def pop(self, k, default=None):
        return dict.pop(self, k, default)


class Ctx:
    """Context manager that proxies widget calls back to the fake module (columns, tabs, expanders...)."""
    def __init__(self, fake):
        self._f = fake

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):
        return getattr(self._f, name)


class Empty(Ctx):
    def markdown(self, body, **kw):
        self._f.out.append(("markdown", body))

    def empty(self):
        pass


class FakeStreamlit(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.session_state = SessionState()
        self.caches = {}
        self.sidebar = Ctx(self)
        self.reset_run()
        self.clicks, self.chat_queue = set(), None
        self.downloads = []

    # ---- run bookkeeping
    def reset_run(self):
        self.out, self.seen_keys = [], set()
        self.registry = {}   # widgets exist only for the run that rendered them

    def _key(self, key, label):
        k = key or f"__label__{label}"
        if k in self.seen_keys:
            raise DuplicateWidgetID(k)
        self.seen_keys.add(k)
        return k

    # ---- non-widget output
    def set_page_config(self, **kw): pass
    def markdown(self, body, **kw): self.out.append(("markdown", body))
    def write(self, body, **kw): self.out.append(("write", str(body)))
    def caption(self, body, **kw): self.out.append(("caption", str(body)))
    def info(self, body, **kw): self.out.append(("info", str(body)))
    def error(self, body, **kw): self.out.append(("error", str(body)))
    def warning(self, body, **kw): self.out.append(("warning", str(body)))
    def success(self, body, **kw): self.out.append(("success", str(body)))
    def toast(self, body, **kw): self.out.append(("toast", str(body)))
    def code(self, body, **kw): self.out.append(("code", str(body)))
    def spinner(self, *a, **k): return Ctx(self)
    def container(self, **k): return Ctx(self)
    def expander(self, *a, **k): return Ctx(self)
    def popover(self, *a, **k): return Ctx(self)
    def empty(self): return Empty(self)
    def columns(self, spec, **k):
        n = spec if isinstance(spec, int) else len(spec)
        return [Ctx(self) for _ in range(n)]
    def tabs(self, labels): return [Ctx(self) for _ in labels]
    def stop(self): raise StopScript()
    def rerun(self): raise RerunException()

    def cache_resource(self, fn=None, **kw):
        return self._cache(fn)
    cache_data = cache_resource

    def _cache(self, fn):
        def wrapper(*a, **k):
            if fn.__name__ not in self.caches:
                self.caches[fn.__name__] = fn(*a, **k)
            return self.caches[fn.__name__]
        return wrapper

    # ---- widgets
    def button(self, label, key=None, on_click=None, args=(), **kw):
        k = self._key(key, label)
        self.registry[k] = {"on_click": on_click, "args": args}
        return k in self.clicks

    def download_button(self, label, data, file_name=None, mime=None, key=None, **kw):
        k = self._key(key, label)
        self.downloads.append((k, file_name, data))
        return False

    def _stateful(self, label, default, key, on_change=None, args=()):
        k = self._key(key, label)
        if key is not None:
            self.registry[k] = {"on_change": on_change, "args": args}
            if k not in self.session_state:
                self.session_state[k] = default
            return self.session_state[k]
        return default

    def checkbox(self, label, value=False, key=None, on_change=None, args=(), **kw):
        return self._stateful(label, value, key, on_change, args)
    toggle = checkbox

    def text_input(self, label, value="", key=None, **kw):
        return self._stateful(label, value, key)

    def radio(self, label, options, index=0, key=None, on_change=None, args=(), **kw):
        return self._stateful(label, options[index], key, on_change, args)

    def multiselect(self, label, options, default=None, key=None, **kw):
        return self._stateful(label, list(default or []), key)

    def chat_input(self, placeholder="", disabled=False, **kw):
        v, self.chat_queue = self.chat_queue, None
        return None if disabled else v


class AppHarness:
    """Run a Streamlit script repeatedly like Streamlit does (rerun on st.rerun / interaction)."""
    def __init__(self, script_path):
        self.script = script_path
        self.st = FakeStreamlit()
        self.exceptions = []
        sys.modules["streamlit"] = self.st
        sys.modules.pop("ui_components", None)   # it binds `st` at import time; re-import against this fake

    def run(self, clicks=(), chat=None):
        st = self.st
        st.clicks, st.chat_queue = set(clicks), chat
        for _ in range(12):
            st.reset_run()
            try:
                runpy.run_path(self.script, run_name="__main__")
            except RerunException:
                st.clicks, st.chat_queue = set(), None
                continue
            except StopScript:
                pass
            break
        else:
            raise RuntimeError("rerun loop did not settle")
        return self

    def click(self, key):
        entry = self.st.registry.get(key)
        assert entry is not None, f"no widget with key {key!r} on the last run"
        if entry.get("on_click"):
            entry["on_click"](*entry["args"])
        return self.run(clicks=[key])

    def set_widget(self, key, value):
        self.st.session_state[key] = value
        entry = self.st.registry.get(key, {})
        if entry.get("on_change"):
            entry["on_change"](*entry["args"])
        return self.run()

    def goto(self, page):
        self.st.session_state["ua_page"] = page
        return self.run()

    def send(self, text):
        return self.run(chat=text)

    @property
    def text(self):
        return "\n".join(f"{kind}: {body}" for kind, body in self.st.out if not body.lstrip().startswith("<style>"))

    def has_key(self, key):
        return key in self.st.registry and any(True for _ in [0])
