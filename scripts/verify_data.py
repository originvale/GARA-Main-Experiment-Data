#!/usr/bin/env python3
"""Verify the released main-experiment records without network access or fitting."""
# SPDX-License-Identifier: Apache-2.0
import csv
import gzip
import hashlib
import heapq
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('Adaptive-RAG', 'MBA-RAG', 'Current-quality', 'GARA')
PIPELINES = {'RAG-QPP': 'rag_qpp', 'Structured Planning': 'structured_planning'}


def readcsv(name):
    with (ROOT / name).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def jsonrows(name):
    with gzip.open(ROOT / name, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def key(row):
    return row['dataset'], row['query_id']


def close(actual, expected):
    actual, expected = float(actual), float(expected)
    assert math.isfinite(actual) and math.isfinite(expected)
    assert abs(actual-expected) < 1e-8, (actual, expected)


def area(curve):
    return sum((a+b)/2 for a, b in zip(curve, curve[1:]))/1800


def verify():
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text(encoding='utf-8'))
    for name, record in manifest['files'].items():
        data = (ROOT / name).read_bytes()
        assert len(data) == record['bytes'], name
        assert hashlib.sha256(data).hexdigest() == record['sha256'], name
    queries = readcsv('queries.csv')
    assert len(queries) == len({key(r) for r in queries}) == 1800
    assert Counter(r['dataset'] for r in queries) == {'2wiki': 600, 'hotpotqa': 600, 'musique': 600}
    identities = {key(r) for r in queries}
    query_map = {key(r): r for r in queries}
    for row in queries:
        assert isinstance(json.loads(row['reference_answers']), list)

    states = readcsv('shared_executor/state_scores.csv')
    state = {(*key(r), int(r['depth'])): r for r in states}
    assert len(states) == len(state) == 5400
    trajectories = list(jsonrows('shared_executor/trajectories.jsonl.gz'))
    assert len(trajectories) == 1800 and {key(r) for r in trajectories} == identities
    for row in trajectories:
        assert row['initial_state']['question'] == query_map[key(row)]['question']
        for depth, entry in enumerate([row['initial_state'], *row['followup_states']]):
            saved = state[(*key(row), depth)]
            assert saved['answer'] == entry['candidate_answer']
            close(saved['F1'], 100*entry['f1'])

    reader = {}
    for row in jsonrows('shared_executor/reader_calls.jsonl.gz'):
        ident = (*key(row), int(row['depth']))
        assert ident not in reader and row['call_id'] == state[ident]['reader_call_id']
        assert row['http_status'] == 200
        assert row['input_tokens'] + row['output_tokens'] == row['total_tokens']
        assert isinstance(row['response_content'], str) and row['response_content']
        reader[ident] = {k: row[k] for k in ('call_id', 'input_tokens', 'output_tokens', 'total_tokens')}
    assert len(reader) == 5400
    retrieved = set()
    for row in jsonrows('shared_executor/retrieval_calls.jsonl.gz'):
        ident = (*key(row), int(row['depth']))
        assert ident in state and ident not in retrieved
        assert row['call_id'] == state[ident]['retrieval_call_id']
        assert row['http_status'] == 200 and row['retrieved_count'] == len(row['passages'])
        retrieved.add(ident)
    assert len(retrieved) == 5400

    priorities = readcsv('allocation/action_values.csv')
    values = {(r['method'], *key(r), int(r['depth'])): float(r['predicted_value']) for r in priorities}
    assert len(values) == len(priorities) == 14400
    records = readcsv('allocation/action_records.csv')
    by_method = defaultdict(list)
    for row in records:
        by_method[row['method']].append(row)
    assert set(by_method) == set(METHODS)
    table2 = {r['method']: r for r in readcsv('paper/table2.csv')}
    figure = defaultdict(list)
    for row in readcsv('paper/figure2_curves.csv'):
        figure[row['method']].append(row)
    final_depths, metric_results = {}, {}
    for method in METHODS:
        assert len(by_method[method]) == 1800
        depths = {q: 0 for q in identities}
        total = sum(float(state[(*q, 0)]['F1']) for q in identities)
        curve = [total/1800]
        heap = [(-values[(method, *q, 1)], *q, 1) for q in identities]
        heapq.heapify(heap)
        for action, row in enumerate(by_method[method], 1):
            q = key(row)
            before, after = int(row['from_depth']), int(row['to_depth'])
            assert int(row['global_action']) == action
            assert depths[q] == before and after == before+1
            minus_score, ds, qid, chosen_depth = heapq.heappop(heap)
            assert (ds, qid, chosen_depth) == (*q, after), (method, action)
            close(row['predicted_value'], -minus_score)
            close(row['F1_before'], state[(*q, before)]['F1'])
            close(row['F1_after'], state[(*q, after)]['F1'])
            total += float(state[(*q, after)]['F1'])-float(state[(*q, before)]['F1'])
            curve.append(total/1800)
            close(row['overall_F1'], curve[-1])
            assert row['reader_call_id'] == reader[(*q, after)]['call_id']
            depths[q] = after
            if after < 2:
                heapq.heappush(heap, (-values[(method, *q, after+1)], *q, after+1))
        assert sum(depths.values()) == 1800
        assert len(figure[method]) == 1801
        for action, row in enumerate(figure[method]):
            assert int(row['actions']) == action
            close(row['budget_ratio'], action/3600)
            close(row['F1'], curve[action])
        result = dict(AUBPC=area(curve), F1_10=curve[360], F1_25=curve[900], F1_50=curve[1800])
        for field, value in result.items():
            close(table2[method][field], value)
        metric_results[method] = result
        final_depths[method] = depths

    terminal = readcsv('allocation/terminal_per_query.csv')
    terminal_groups = defaultdict(list)
    assert len(terminal) == 7200
    for row in terminal:
        method, q, depth = row['method'], key(row), int(row['depth'])
        assert final_depths[method][q] == depth
        assert row['answer'] == state[(*q, depth)]['answer']
        close(row['F1'], state[(*q, depth)]['F1'])
        close(row['EM'], state[(*q, depth)]['EM'])
        assert int(row['calls']) == depth+1
        assert int(row['total_tokens']) == sum(reader[(*q, d)]['total_tokens'] for d in range(depth+1))
        row['retrievals'] = row['calls']
        terminal_groups[method].append(row)

    pipeline_terminal = readcsv('pipelines/terminal_per_query.csv')
    assert len(pipeline_terminal) == 3600
    dependency = defaultdict(list)
    for row in readcsv('pipelines/action_records.csv'):
        dependency[(row['method'], *key(row))].append(row)
    for method, folder in PIPELINES.items():
        calls, retrievals, groups = {}, {}, defaultdict(list)
        for row in jsonrows(f'pipelines/{folder}/reader_calls.jsonl.gz'):
            assert row['call_id'] not in calls and key(row) in identities
            assert row['http_status'] == 200
            assert row['input_tokens'] + row['output_tokens'] == row['total_tokens']
            assert row['response_content'] == row['response_message']['content']
            calls[row['call_id']] = {k: row[k] for k in ('input_tokens', 'output_tokens', 'total_tokens')}
            groups[key(row)].append(row['call_id'])
        for row in jsonrows(f'pipelines/{folder}/retrieval_calls.jsonl.gz'):
            assert row['call_id'] not in retrievals and key(row) in identities
            assert row['requested_depth'] == len(row['passages'])
            retrievals[row['call_id']] = key(row)
        trajectories = list(jsonrows(f'pipelines/{folder}/trajectories.jsonl.gz'))
        assert len(trajectories) == 1800 and {key(r) for r in trajectories} == identities
        tm = {key(r): r for r in pipeline_terminal if r['method'] == method}
        assert set(tm) == identities
        for trajectory in trajectories:
            q = key(trajectory)
            result = tm[q]
            ids = trajectory['reader_call_ids']
            assert set(ids) == set(groups[q]) and len(ids) == int(result['calls'])
            assert len(trajectory['retrieval_call_ids']) == int(result['retrievals'])
            assert all(retrievals[i] == q for i in trajectory['retrieval_call_ids'])
            assert result['answer'] == trajectory['answer']
            for field in ('input_tokens', 'output_tokens', 'total_tokens'):
                assert int(result[field]) == sum(calls[i][field] for i in ids)
            events = dependency[(method, *q)]
            assert [int(r['event_index']) for r in events] == list(range(1, len(events)+1))
            assert [r['call_id'] for r in events if r['kind'] == 'reader'] == ids
            assert [r['call_id'] for r in events if r['kind'] == 'retrieval'] == trajectory['retrieval_call_ids']
            terminal_groups[method].append(result)

    table3 = {r['method']: r for r in readcsv('paper/table3.csv')}
    for method, group in terminal_groups.items():
        assert len(group) == 1800 and {key(r) for r in group} == identities
        expected = table3[method]
        for ds in ('2wiki', 'hotpotqa', 'musique'):
            subset = [r for r in group if r['dataset'] == ds]
            assert len(subset) == 600
            close(expected[ds+'_F1'], sum(float(r['F1']) for r in subset)/600)
        for metric in ('F1', 'EM'):
            close(expected['Overall_'+metric], sum(float(r[metric]) for r in group)/1800)
        close(expected['LLM_calls_per_query'], sum(int(r['calls']) for r in group)/1800)
        close(expected['retrievals_per_query'], sum(int(r['retrievals']) for r in group)/1800)
        close(expected['total_LLM_tokens'], sum(int(r['total_tokens']) for r in group))

    oracle = readcsv('offline_oracle/curve.csv')
    assert len(oracle) == 1801
    for i, row in enumerate(oracle):
        assert int(row['actions']) == i
        close(row['budget_ratio'], i/3600)
        assert all(float(row['Oracle_F1'])+1e-8 >= float(figure[m][i]['F1']) for m in METHODS)
    close(table2['Offline Oracle']['AUBPC'], area([float(r['Oracle_F1']) for r in oracle]))
    for budget, field in ((360, 'F1_10'), (900, 'F1_25'), (1800, 'F1_50')):
        allocation = readcsv(f'offline_oracle/allocation_B{budget}.csv')
        assert len(allocation) == 1800 and {key(r) for r in allocation} == identities
        assert sum(int(r['followups']) for r in allocation) == budget
        for row in allocation:
            close(row['F1'], state[(*key(row), int(row['followups']))]['F1'])
        mean = sum(float(r['F1']) for r in allocation)/1800
        close(table2['Offline Oracle'][field], mean)
        close(oracle[budget]['Oracle_F1'], mean)
    return dict(status='PASS', queries=1800, tables=[2, 3],
                verified_files=len(manifest['files']), main_metrics=metric_results)


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
