"""真实训练入口前的监督位置检查，不用loss下降代替任务质量。"""

import importlib
import importlib.util
import pytest


def test_only_action_and_eos_receive_loss_and_no_silent_truncation():
    assert importlib.util.find_spec('serving_lab.coldstart_train') is not None
    encode = importlib.import_module('serving_lab.coldstart_train').encode_example

    class Tokenizer:
        eos_token_id = 9

        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{'role': 'user', 'content': 'public prefix'}]
            assert kwargs['enable_thinking'] is False
            return [1, 2, 3]

        def encode(self, text, **kwargs):
            assert text == 'look'
            return [4, 5]

    row = {
        'messages': [
            {'role': 'user', 'content': 'public prefix'},
            {'role': 'assistant', 'content': 'look'},
        ]
    }
    assert encode(row, Tokenizer(), max_length=6) == (
        [1, 2, 3, 4, 5, 9],
        [-100, -100, -100, 4, 5, 9],
        [1, 2, 3],
    )
    with pytest.raises(ValueError):
        encode(row, Tokenizer(), max_length=5)


def test_target_only_projection_preserves_full_causal_loss_and_gradients():
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    import copy
    from serving_lab import coldstart_train as c

    assert hasattr(c, 'supervised_tail')
    labels = [-100, -100, -100, -100, 9, 10, 2]
    keep, tail = c.supervised_tail(labels)
    assert keep == 4 and tail == [-100, 9, 10, 2]
    torch.manual_seed(7)
    config = transformers.Qwen3Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    full = transformers.Qwen3ForCausalLM(config).eval()
    efficient = copy.deepcopy(full)
    x = torch.tensor([[1, 5, 8, 3, 9, 10, 2]])
    a = full(input_ids=x, labels=torch.tensor([labels]), use_cache=False).loss
    b = efficient(
        input_ids=x, labels=torch.tensor([tail]), logits_to_keep=keep, use_cache=False
    ).loss
    torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
    a.backward()
    b.backward()
    for left, right in zip(full.parameters(), efficient.parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-6)
