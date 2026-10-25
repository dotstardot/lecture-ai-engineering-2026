"""第1回: 会話履歴を保ちながらLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

起動時に、APIで利用可能なモデルから会話に使うモデルを選択する。

ツールを持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import time

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def choose_model(client: genai.Client) -> str:
    """ListModels で取得した generateContent 対応モデルから1つ選ぶ。"""
    models = [
        model
        for model in client.models.list()
        if "generateContent" in (model.supported_actions or [])
    ]
    if not models:
        raise RuntimeError("generateContent に対応したモデルが見つかりませんでした")

    model_names = [model.name.removeprefix("models/") for model in models]
    default_index = next(
        (index for index, name in enumerate(model_names) if name == MODEL.removeprefix("models/")),
        0,
    )

    print("利用可能なモデル:")
    for index, name in enumerate(model_names, start=1):
        marker = " (既定)" if index - 1 == default_index else ""
        print(f"  {index}. {name}{marker}")

    while True:
        selection = input(f"使用するモデル番号 [{default_index + 1}]: ").strip()
        if not selection:
            return model_names[default_index]
        try:
            index = int(selection) - 1
        except ValueError:
            index = -1
        if 0 <= index < len(model_names):
            return model_names[index]
        print(f"1〜{len(model_names)}の番号を入力してください。")


def ask(question: str, chat=None) -> str:
    if chat is None:
        client = genai.Client(api_key=get_api_key())
        chat = client.chats.create(model=MODEL)
    started_at = time.perf_counter()
    response = chat.send_message(question)
    elapsed = time.perf_counter() - started_at

    usage = response.usage_metadata
    print(
        f"[usage] input={usage.prompt_token_count} "
        f"output={usage.candidates_token_count} elapsed={elapsed:.3f}s"
    )

    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    client = genai.Client(api_key=get_api_key())
    question = args.question
    try:
        selected_model = choose_model(client)
        chat = client.chats.create(model=selected_model)
        print(f"[model] {selected_model}")
        while question:
            print(f"[質問] {question}\n")
            print(ask(question, chat))
            try:
                question = input("\n続けて質問（終了するには空行または Ctrl-D）: ").strip()
            except EOFError:
                break
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
