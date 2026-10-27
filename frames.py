"""FRAMES evidence, answers, groundedness judging, and registered D19 checks."""
import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics as st
import time
import urllib.error
import urllib.parse
import urllib.request

from lens.evidence import evidence, count, _raw
from lens.pick import HAIKU, _cached, _claude

SONNET = 'claude-sonnet-5-5'
ARMS = ('raw', 'rag', 'lens', 'rerank')
ROOT = Path('data/frames')
USER_AGENT = 'lens-research/0.1 (github.com/SrujanPendurthi/lens)'
ANSWER_SYSTEM = 'Answer the question using only the evidence provided. Be concise. If the evidence is not enough, reply exactly: I could not find this.'
JUDGE_SYSTEM = 'You grade answers for groundedness. SOURCE PAGES are the complete pages the answers could use. For each ANSWER: split it into atomic factual claims; label each claim SUPPORTED if the source pages state or directly entail it, else UNSUPPORTED; set verdict CORRECT if it matches the GOLD ANSWER in substance, else INCORRECT. An answer saying it could not find the information has no claims and verdict ABSTAINED. Reply with JSON only: {"answers": [{"id": "a1", "claims": [{"claim": "...", "label": "SUPPORTED"}], "verdict": "CORRECT"}]}'
_last_request = 0.


def abstained(text):
    return text.strip().lower().startswith('i could not find this')


def answer(question, source):
    prompt = f'Evidence:\n{source}\n\nQuestion: {question}'
    key = hashlib.sha1(f'frames-answer-v1\n{HAIKU}\n{prompt}'.encode()).hexdigest()
    result = _cached(key, lambda: _claude(HAIKU, ANSWER_SYSTEM, prompt))
    return result | {'abstained': abstained(result['out']), 'claims': []}


def judge(q, pages, answers):
    order = list(ARMS)
    random.Random(q['id']).shuffle(order)
    mapping = {f'a{i+1}': arm for i, arm in enumerate(order)}
    sources, _, _ = _raw(pages, 150000, style=True)
    prompt = f'SOURCE PAGES:\n{sources}\n\nQUESTION: {q["question"]}\nGOLD ANSWER: {q["answer"]}\n\n'
    prompt += '\n\n'.join(f'ANSWER {identifier}:\n{answers[arm]}' for identifier, arm in mapping.items())
    key = hashlib.sha1(f'frames-judge-v1\n{SONNET}\n{prompt}'.encode()).hexdigest()

    def generate():
        response = _claude(SONNET, JUDGE_SYSTEM, prompt, timeout=600)
        data = json.loads(response['out'])
        entries = data['answers']
        if len(entries) != 4 or {r['id'] for r in entries} != set(mapping):
            raise ValueError('judge must return every anonymized answer exactly once')
        for row in entries:
            if row['verdict'] not in ('CORRECT', 'INCORRECT', 'ABSTAINED'):
                raise ValueError('invalid judge verdict')
            for claim in row['claims']:
                if not isinstance(claim['claim'], str) or claim['label'] not in ('SUPPORTED', 'UNSUPPORTED'):
                    raise ValueError('invalid atomic claim')
        return data

    result = {}
    for entry in _cached(key, generate)['answers']:
        arm = mapping[entry['id']]
        is_abstained = abstained(answers[arm])
        result[arm] = {'claims': [] if is_abstained else entry['claims'],
                       'verdict': 'ABSTAINED' if is_abstained else entry['verdict'],
                       'abstained': is_abstained}
    return result


def unsupported(row):
    return sum(c['label'] == 'UNSUPPORTED' for c in row['claims'])


def gate(rows, c, b):
    pairs = [(r['arms'][c], r['arms'][b]) for r in rows]
    wins = sum(unsupported(a) < unsupported(z) for a, z in pairs)
    losses = sum(unsupported(a) > unsupported(z) for a, z in pairs)
    decided = wins + losses
    p = sum(math.comb(decided, k) for k in range(wins, decided+1))/2**decided if decided else 1.
    def pooled(arm):
        claims = sum(len(r['arms'][arm]['claims']) for r in rows)
        return sum(unsupported(r['arms'][arm]) for r in rows)/claims if claims else 0.
    failures = []
    if wins-losses < 5: failures.append('net')
    if p >= .05: failures.append('significance')
    if pooled(c) >= pooled(b): failures.append('unsupported rate')
    if (sum(a['abstained'] for a, _ in pairs)-sum(z['abstained'] for _, z in pairs))*10 > len(rows):
        failures.append('abstention')
    if st.median(a['ms'] for a, _ in pairs) > 1.25*st.median(z['ms'] for _, z in pairs):
        failures.append('speed')
    verdict = 'FAIL (' + ', '.join(failures) + ')' if failures else 'PASS'
    return f'wins {wins}, losses {losses}, net {wins-losses:+d}, p={p:.3f} → {verdict}'


def summarize(rows):
    print(f'questions={len(rows)}')
    print('arm claims unsupported unsupported% abstain% accuracy% evidence_tokens prompt_tokens prompt_minus203 retrieval_ms')
    for arm in ARMS:
        values = [r['arms'][arm] for r in rows]
        claims = sum(len(v['claims']) for v in values)
        bad = sum(map(unsupported, values))
        print(f'{arm} {claims} {bad} {100*bad/claims if claims else 0:.2f} '
              f'{100*sum(v["abstained"] for v in values)/len(rows):.1f} '
              f'{100*sum(v["verdict"]=="CORRECT" for v in values)/len(rows):.1f} '
              f'{st.median(v["evidence_tokens"] for v in values):.0f} '
              f'{st.median(v["prompt_tokens"] for v in values):.0f} '
              f'{st.median(v["prompt_tokens"]-203 for v in values):.0f} '
              f'{st.median(v["ms"] for v in values):.2f}')
    print('D19 lens vs rag: ' + gate(rows, 'lens', 'rag'))
    print('D19 product edge lens vs rerank: ' + gate(rows, 'lens', 'rerank'))
    print('reported rerank vs rag: ' + gate(rows, 'rerank', 'rag'))
    print(f'raw truncated: {sum(r["arms"]["raw"]["truncated"] for r in rows)}/{len(rows)}')


def page_path(url):
    return ROOT/'pages'/(hashlib.sha1(url.encode()).hexdigest()+'.json')


def fetch_page(url):
    global _last_request
    saved = page_path(url)
    if saved.exists():
        return json.loads(saved.read_text())
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != 'https' or not re.fullmatch(r'[a-z0-9-]+(?:\.m)?\.wikipedia\.org', parsed.hostname or '') or not parsed.path.startswith('/wiki/'):
        raise ValueError(f'unsupported Wikipedia URL: {url}')
    title = urllib.parse.unquote(parsed.path[len('/wiki/'):])
    query = urllib.parse.urlencode({'action': 'query', 'format': 'json', 'formatversion': 2,
                                  'prop': 'extracts|revisions', 'explaintext': 1, 'rvprop': 'ids|timestamp',
                                  'redirects': 1, 'titles': title})
    api_host = parsed.hostname.replace('.m.wikipedia.org', '.wikipedia.org')
    request = urllib.request.Request(f'https://{api_host}/w/api.php?'+query,
                                     headers={'User-Agent': USER_AGENT})
    for attempt in range(4):
        delay = max(0., 1.-(time.monotonic()-_last_request))
        if delay: time.sleep(delay)
        _last_request = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = json.load(response)
            break
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == 3:
                raise
            retry_after = exc.headers.get('Retry-After', '') if exc.headers else ''
            try:
                wait = max(1., float(retry_after))
            except ValueError:
                wait = 60. * (attempt + 1)
            print(f'Wikipedia HTTP 429; retry {attempt+1}/3 in {wait:.0f}s: {url}', flush=True)
            time.sleep(wait)
    page = data['query']['pages'][0]
    if page.get('missing') or not page.get('extract', '').strip():
        raise ValueError(f'no page extract: {url}')
    revision = page['revisions'][0]
    result = {'url': url, 'title': page['title'], 'revid': revision['revid'],
              'timestamp': revision['timestamp'], 'text': page['extract']}
    saved.parent.mkdir(parents=True, exist_ok=True)
    saved.write_text(json.dumps(result))
    return result


def prep():
    from huggingface_hub import hf_hub_download
    ROOT.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download('google/frames-benchmark', 'test.tsv', repo_type='dataset', local_dir=str(ROOT))
    with open(path) as stream:
        source = list(csv.DictReader(stream, delimiter='\t'))
    if len(source) != 824:
        raise ValueError(f'expected 824 FRAMES rows, found {len(source)}')
    rng = random.Random(0)
    initial = rng.sample(range(len(source)), 100)
    remaining = [i for i in range(len(source)) if i not in set(initial)]
    rng.shuffle(remaining)
    reserve = iter(remaining)
    selected, replacements = [], []
    for original in initial:
        index = original
        while True:
            row = source[index]
            q = {'id': int(row.get('') or index), 'question': row['Prompt'], 'answer': row['Answer'],
                 'pages': ast.literal_eval(row['wiki_links'])}
            try:
                for url in q['pages']: fetch_page(url)
            except (urllib.error.URLError, ValueError, KeyError) as exc:
                replacements.append({'id': q['id'], 'reason': str(exc)})
                (ROOT/'replacements.json').write_text(json.dumps(replacements, indent=2))
                index = next(reserve)
                continue
            selected.append(q)
            print(f'prepared questions: {len(selected)}/100', flush=True)
            break
    with (ROOT/'questions.jsonl').open('w') as stream:
        for q in selected: stream.write(json.dumps(q)+'\n')
    cached = {url: fetch_page(url) for q in selected for url in q['pages']}
    tokens = [count('\n\n'.join(f'[{cached[u]["title"]}]\n{cached[u]["text"]}' for u in q['pages'])) for q in selected]
    timestamps = [p['timestamp'] for p in cached.values()]
    stats = {'questions': len(selected), 'pages': len(cached), 'median_tokens': st.median(tokens),
             'max_tokens': max(tokens), 'replacements': replacements,
             'snapshot_date_range': [min(timestamps), max(timestamps)]}
    (ROOT/'replacements.json').write_text(json.dumps(replacements, indent=2))
    (ROOT/'snapshot.json').write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2), flush=True)


def run(n, output):
    questions = [json.loads(line) for line in (ROOT/'questions.jsonl').open()][:n]
    if len(questions) != n:
        raise ValueError('not enough prepared questions')
    rows = []
    with open(output, 'w') as stream:
        for q in questions:
            pages = {}
            for url in q['pages']:
                p = json.loads(page_path(url).read_text())
                pages[p['title']] = p['text']
            arms = {}
            for arm in ARMS:
                text, stats = evidence(arm, q['question'], pages)
                cap = 150000 if arm == 'raw' else 2000
                if stats['tokens'] > cap:
                    raise ValueError(f'{arm} exceeded its token cap')
                response = answer(q['question'], text)
                arms[arm] = {'answer': response['out'], 'abstained': response['abstained'],
                             'evidence_tokens': stats['tokens'], 'prompt_tokens': response['prompt_tokens'],
                             'ms': stats['ms'], 'truncated': stats['truncated']}
            labels = judge(q, pages, {arm: a['answer'] for arm, a in arms.items()})
            for arm in ARMS: arms[arm].update(labels[arm])
            row = {'id': q['id'], 'arms': arms}
            rows.append(row)
            stream.write(json.dumps(row)+'\n');stream.flush()
            print(f'completed questions: {len(rows)}/{n}', flush=True)
    summarize(rows)


def judge_sample(path):
    rows = [json.loads(line) for line in Path(path).open()][:10]
    questions = {q['id']: q for q in map(json.loads, (ROOT/'questions.jsonl').open())}
    claims = []
    for row in rows:
        q = questions[row['id']]
        titles = [json.loads(page_path(url).read_text())['title'] for url in q['pages']]
        for arm in ARMS:
            for claim in row['arms'][arm]['claims']:
                claims.append({'id': row['id'], 'question': q['question'], 'page_titles': titles,
                               'claim': claim['claim'], 'sonnet_label': claim['label'],
                               'opus_label': None, 'final_label': None})
    sampled = random.Random(0).sample(claims, 30)
    with open('frames_judge_check.jsonl', 'w') as stream:
        for claim in sampled: stream.write(json.dumps(claim)+'\n')
    print('judge sample: 30 claims → frames_judge_check.jsonl')


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prep', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--summarize')
    mode.add_argument('--judge-sample')
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--output', default='results_frames.jsonl')
    args = parser.parse_args()
    if args.prep: prep()
    elif args.run: run(args.n, args.output)
    elif args.summarize:
        rows = [json.loads(line) for line in Path(args.summarize).open()]
        if not rows: parser.error('no rows to summarize')
        summarize(rows)
    else: judge_sample(args.judge_sample)


if __name__ == '__main__':
    main()
