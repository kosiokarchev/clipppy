from pyro.poutine import NonlocalExit
from pyro.poutine.escape_messenger import EscapeMessenger


class PostEscapeMessenger(EscapeMessenger):
    def _pyro_sample(self, msg):
        return None

    def _pyro_post_sample(self, msg):
        return super()._pyro_sample(msg)

    def __exit__(self, exc_type, exc_val, exc_tb):
        return True if (
            ret := super().__exit__(exc_type, exc_val, exc_tb)
        ) is None and exc_type is NonlocalExit else ret


class UpToMessenger(PostEscapeMessenger):
    def escape_fn(self, msg):
        self.seen.add(msg['name'])
        return not bool(self.names - self.seen)

    def __init__(self, *names: str):
        super().__init__(self.escape_fn)
        self.names = set(names)
        self.seen = set()


class CollectSitesMessenger(UpToMessenger, dict):
    def escape_fn(self, msg):
        if msg['name'] in self.names:
            self[msg['name']] = msg
        return super().escape_fn(msg)


# class CollectSitesMessenger(PostEscapeMessenger, dict):
#     def escape_fn(self, msg):
#         if msg['name'] in self.names:
#             self[msg['name']] = msg['value']
#         return not bool(self.names - self.keys())
#
#     def __init__(self, *names: str):
#         super().__init__(self.escape_fn)
#         self.names = set(names)

