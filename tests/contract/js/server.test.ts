// The pinned JS SDK against a real gateway (fake engine). Started by tests/contract/test_server_sdk.py, which sets
// GATEWAY_TEST_BASE_URL, GATEWAY_TEST_API_KEY and GATEWAY_TEST_AUDIO_DIR (speech.wav, fail.wav, turn*.pcm, expected.json),
// plus GATEWAY_TEST_WSS_BASE_URL and GATEWAY_TEST_CA (a TLS gateway: OpenAIRealtimeWS always connects with wss:);
// without them (plain `npm test`) every test here is skipped.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { test } from 'node:test';
import OpenAI, { toFile } from 'openai';
import { OpenAIRealtimeWS } from 'openai/realtime/ws';
import type { OpenAIRealtimeError } from 'openai/realtime/internal-base';
import type { TranscriptionStreamEvent } from 'openai/resources/audio/transcriptions';
import type { RealtimeServerEvent } from 'openai/resources/realtime/realtime';

const baseURL = process.env.GATEWAY_TEST_BASE_URL;
const apiKey = process.env.GATEWAY_TEST_API_KEY;
const audioDir = process.env.GATEWAY_TEST_AUDIO_DIR ?? '';
const skip = baseURL && apiKey && audioDir ? false : 'no gateway: run tests/contract/test_server_sdk.py';
const MODEL = 'viet-asr';

const examples = new URL('../examples/', import.meta.url);
const example = (name: string) => JSON.parse(readFileSync(new URL(name, examples)).toString());
const client = () => new OpenAI({ apiKey, baseURL, maxRetries: 0 });
const audio = (name: string) => toFile(readFileSync(join(audioDir, name)), name);
type Expected = { text: string; deltas: string[]; seconds: number; turns: { deltas: string[]; seconds: number }[] };
const expected = (): Expected => JSON.parse(readFileSync(join(audioDir, 'expected.json')).toString());

test('models', { skip }, async () => {
  const page = await client().models.list();
  assert.deepEqual(page.data, example('models.json').data);
  assert.deepEqual(await client().models.retrieve(MODEL), example('model.json'));
  await assert.rejects(
    client().models.retrieve('whisper-1'),
    (err: unknown) => err instanceof OpenAI.NotFoundError && err.code === 'model_not_found',
  );
});

test('json and text transcription', { skip }, async () => {
  const exp = expected();
  const t = await client().audio.transcriptions.create({
    file: await audio('speech.wav'),
    model: MODEL,
    language: 'vi',
    temperature: 0,
  });
  assert.deepEqual(t, { text: exp.text, usage: { type: 'duration', seconds: exp.seconds } });
  const text = await client().audio.transcriptions.create({
    file: await audio('speech.wav'),
    model: MODEL,
    response_format: 'text',
  });
  assert.equal(text, exp.text);
});

// Cue texts of a gateway SRT/VTT body (the last line of each block).
const cueTexts = (body: string) =>
  body.replace(/^WEBVTT\n\n/, '').split('\n\n').filter((b) => b).map((b) => b.split('\n').at(-1));

test('verbose_json, srt and vtt carry the same text', { skip }, async () => {
  const exp = expected();
  const v = await client().audio.transcriptions.create({
    file: await audio('speech.wav'),
    model: MODEL,
    response_format: 'verbose_json',
  });
  assert.deepEqual(v, {
    text: exp.text,
    language: 'vietnamese',
    duration: exp.seconds,
    usage: { type: 'duration', seconds: exp.seconds },
  });
  for (const fmt of ['srt', 'vtt'] as const) {
    const body: string = await client().audio.transcriptions.create({
      file: await audio('speech.wav'),
      model: MODEL,
      response_format: fmt,
    });
    assert.equal(cueTexts(body).join(' '), exp.text);
    assert.ok(fmt === 'srt' ? body.startsWith('1\n') : body.startsWith('WEBVTT\n\n'));
  }
  await assert.rejects(
    client().audio.transcriptions.create({
      file: await audio('speech.wav'),
      model: MODEL,
      response_format: 'verbose_json',
      timestamp_granularities: ['word'],
    }),
    (err: unknown) =>
      err instanceof OpenAI.BadRequestError &&
      err.code === 'unsupported_parameter' &&
      err.param === 'timestamp_granularities[]',
  );
});

test('SSE stream: deltas join to done', { skip }, async () => {
  const exp = expected();
  const stream = await client().audio.transcriptions.create({ file: await audio('speech.wav'), model: MODEL, stream: true });
  const events: TranscriptionStreamEvent[] = [];
  for await (const e of stream) events.push(e);
  const deltas = events.flatMap((e) => (e.type === 'transcript.text.delta' ? [e.delta] : []));
  assert.deepEqual(deltas, exp.deltas);
  const done = events.at(-1)!;
  assert.ok(done.type === 'transcript.text.done' && done.text === exp.text && done.usage === undefined);
  assert.equal(events.length, deltas.length + 1);
});

test('SSE error after headers rejects without done', { skip }, async () => {
  const stream = await client().audio.transcriptions.create({ file: await audio('fail.wav'), model: MODEL, stream: true });
  const got: string[] = [];
  await assert.rejects(
    (async () => {
      for await (const e of stream) got.push(e.type);
    })(),
    (err: unknown) => err instanceof OpenAI.APIError && err.code === 'inference_failed' && err.type === 'server_error',
  );
  assert.ok(got.length >= 1 && got.every((t) => t === 'transcript.text.delta'));
});

test('HTTP error envelope and request id', { skip }, async () => {
  await assert.rejects(
    client().audio.transcriptions.create({ file: await audio('speech.wav'), model: MODEL, language: 'en' }),
    (err: unknown) =>
      err instanceof OpenAI.BadRequestError &&
      err.code === 'unsupported_language' &&
      err.param === 'language' &&
      (err.requestID ?? '').startsWith('req_'),
  );
});

const wssBaseURL = process.env.GATEWAY_TEST_WSS_BASE_URL;
const ca = process.env.GATEWAY_TEST_CA;
const rtSkip = wssBaseURL && ca && !skip ? false : 'no TLS gateway: run tests/contract/test_server_sdk.py';
const rtClient = (key = apiKey) => new OpenAI({ apiKey: key, baseURL: wssBaseURL, maxRetries: 0 });
const tls = () => ({ ca: readFileSync(ca!) });

// Records every server event; `until(type)` returns the events since the last call, up to the next `type`.
function recorder(rt: OpenAIRealtimeWS) {
  const events: RealtimeServerEvent[] = [];
  const errors: OpenAIRealtimeError[] = [];
  let cursor = 0;
  rt.on('event', (e) => events.push(e));
  rt.on('error', (e) => errors.push(e));
  async function until(type: RealtimeServerEvent['type']): Promise<RealtimeServerEvent[]> {
    const deadline = Date.now() + 5000;
    for (;;) {
      const i = events.findIndex((e, j) => j >= cursor && e.type === type);
      if (i >= 0) {
        const got = events.slice(cursor, i + 1);
        cursor = i + 1;
        return got;
      }
      assert.ok(Date.now() < deadline, `timed out waiting for ${type}`);
      await new Promise((r) => setTimeout(r, 10));
    }
  }
  return { events, errors, until };
}

test('realtime over wss: commits, clear and errors', { skip: rtSkip }, async () => {
  const exp = expected();
  const rt = new OpenAIRealtimeWS({ intent: 'transcription', options: tls() }, rtClient());
  const r = recorder(rt);
  const [created] = await r.until('session.created');
  assert.ok(created?.type === 'session.created');
  assert.ok(rt.url.protocol === 'wss:' && rt.url.searchParams.get('intent') === 'transcription');
  rt.send({ type: 'session.update', session: { type: 'transcription', audio: { input: { turn_detection: null } } } });
  const [updated] = await r.until('session.updated');
  assert.deepEqual(updated?.type === 'session.updated' && updated.session, created.session);

  let previous: string | null = null;
  for (const [i, turn] of exp.turns.entries()) {
    const pcm = readFileSync(join(audioDir, `turn${i}.pcm`));
    for (let j = 0; j < pcm.length; j += 4801) {
      rt.send({ type: 'input_audio_buffer.append', audio: pcm.subarray(j, j + 4801).toString('base64') });
    }
    rt.send({ type: 'input_audio_buffer.commit' });
    const [committed, added, ...rest] = await r.until('conversation.item.input_audio_transcription.completed');
    assert.ok(committed?.type === 'input_audio_buffer.committed' && added?.type === 'conversation.item.added');
    assert.equal(committed.previous_item_id, previous);
    assert.equal(added.item.id, committed.item_id);
    const done = rest.pop();
    assert.ok(done?.type === 'conversation.item.input_audio_transcription.completed');
    const deltas = rest.map((e) => (e.type === 'conversation.item.input_audio_transcription.delta' ? e.delta : '?'));
    assert.deepEqual(deltas, turn.deltas);
    assert.equal(done.transcript, turn.deltas.join(''));
    assert.deepEqual(done.usage, { type: 'duration', seconds: turn.seconds });
    previous = committed.item_id;
  }
  rt.send({ type: 'input_audio_buffer.append', audio: Buffer.alloc(960).toString('base64') });
  rt.send({ type: 'input_audio_buffer.clear' });
  await r.until('input_audio_buffer.cleared');
  rt.send({ type: 'input_audio_buffer.commit', event_id: 'client_evt_7' });
  const [err] = await r.until('error');
  assert.ok(err?.type === 'error' && err.error.code === 'input_audio_buffer_commit_empty');
  assert.equal(err.error.event_id, 'client_evt_7');
  assert.equal(r.errors.length, 1); // the SDK surfaces error events as OpenAIRealtimeError
  assert.equal(r.errors[0]?.error?.code, 'input_audio_buffer_commit_empty');
  const ids = r.events.map((e) => ('event_id' in e ? e.event_id : ''));
  assert.ok(ids.every((id) => id.startsWith('event_')));
  assert.equal(new Set(ids).size, ids.length);
  rt.close();
});

test('realtime handshake errors reach the SDK', { skip: rtSkip }, async () => {
  for (const [rt, status] of [
    [new OpenAIRealtimeWS({ model: 'whisper-1', options: tls() }, rtClient()), 404],
    [new OpenAIRealtimeWS({ intent: 'transcription', options: tls() }, rtClient('wrong-key-0123456789')), 401],
  ] as const) {
    const error = await new Promise<OpenAIRealtimeError>((resolve) => rt.on('error', resolve));
    assert.match(error.message, new RegExp(`Unexpected server response: ${status}`));
  }
});
