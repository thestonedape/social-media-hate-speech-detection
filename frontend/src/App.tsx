import { useEffect, useState } from 'react';
import Alert from '@mui/material/Alert';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Chip from '@mui/material/Chip';
import CircularProgress from '@mui/material/CircularProgress';
import Container from '@mui/material/Container';
import Divider from '@mui/material/Divider';
import LinearProgress from '@mui/material/LinearProgress';
import Stack from '@mui/material/Stack';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import ForumOutlinedIcon from '@mui/icons-material/ForumOutlined';
import GavelOutlinedIcon from '@mui/icons-material/GavelOutlined';
import ReportGmailerrorredOutlinedIcon from '@mui/icons-material/ReportGmailerrorredOutlined';
import SentimentSatisfiedAltOutlinedIcon from '@mui/icons-material/SentimentSatisfiedAltOutlined';
import WarningAmberOutlinedIcon from '@mui/icons-material/WarningAmberOutlined';

type Label = 'hate' | 'offensive' | 'normal';

type PredictionResponse = {
  text: string;
  label: Label;
  confidence: number;
  probabilities: Record<Label, number>;
  model_backend: string;
};

type HealthResponse = {
  status: string;
  model_backend: string;
  labels: Label[];
};

const API_URL = (
  import.meta.env.VITE_API_URL ?? 'http://localhost:8000'
).replace(/\/+$/, '');
const MAX_TEXT_CHARS = 2000;
const REQUEST_TIMEOUT_MS = 30_000;

if (
  import.meta.env.PROD &&
  (API_URL.includes('localhost') || API_URL.includes('127.0.0.1'))
) {
  console.error(
    'Production build is using a local API URL. Set VITE_API_URL in Vercel.',
  );
}

const LABEL_META: Record<
  Label,
  { title: string; severity: 'error' | 'warning' | 'success'; blurb: string }
> = {
  hate: {
    title: 'Hate speech',
    severity: 'error',
    blurb: 'Targets a protected group or identity with hostility.',
  },
  offensive: {
    title: 'Offensive',
    severity: 'warning',
    blurb: 'Abusive or profane language without a protected-group target.',
  },
  normal: {
    title: 'Normal',
    severity: 'success',
    blurb: 'No abusive or hateful content detected.',
  },
};

const SAMPLE_TEXTS = [
  'Had a great time at the community meetup today, everyone was so welcoming!',
  'You are such an idiot, nobody wants your dumb opinions here.',
];

function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function ProbabilityRow({
  label,
  value,
  color,
}: {
  label: string;
  value: number | null;
  color: 'error' | 'warning' | 'success';
}) {
  return (
    <Box>
      <Stack
        direction="row"
        sx={{ justifyContent: 'space-between', mb: 0.5 }}
      >
        <Typography variant="body2" color="text.secondary">
          {label}
        </Typography>
        <Typography variant="body2" sx={{ fontWeight: 600 }}>
          {value === null ? '--' : formatPercent(value)}
        </Typography>
      </Stack>
      <LinearProgress
        variant="determinate"
        color={color}
        value={value === null ? 0 : value * 100}
      />
    </Box>
  );
}

export default function App() {
  const [text, setText] = useState('');
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [backendStatus, setBackendStatus] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 15_000);

    void fetch(`${API_URL}/health`, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) return;
        const payload = (await response.json()) as HealthResponse;
        setBackendStatus(payload.model_backend);
      })
      .catch(() => undefined)
      .finally(() => window.clearTimeout(timeout));

    return () => {
      window.clearTimeout(timeout);
      controller.abort();
    };
  }, []);

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = text.trim();
    if (!trimmed) {
      setError('Enter some text to analyze first.');
      return;
    }

    setLoading(true);
    setError(null);
    setPrediction(null);
    const controller = new AbortController();
    const timeout = window.setTimeout(
      () => controller.abort(),
      REQUEST_TIMEOUT_MS,
    );

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: trimmed }),
        signal: controller.signal,
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        const detail = payload?.detail;
        throw new Error(
          typeof detail === 'string' ? detail : 'Prediction request failed.',
        );
      }

      const data = (await response.json()) as PredictionResponse;
      setPrediction(data);
      setBackendStatus(data.model_backend);
    } catch (submissionError) {
      setError(
        submissionError instanceof DOMException &&
          submissionError.name === 'AbortError'
          ? 'The API took too long to respond. It may be waking from sleep; try again.'
          : submissionError instanceof Error
          ? submissionError.message
          : 'Unexpected error.',
      );
    } finally {
      window.clearTimeout(timeout);
      setLoading(false);
    }
  }

  const meta = prediction ? LABEL_META[prediction.label] : null;

  return (
    <Box
      sx={{
        minHeight: '100vh',
        background:
          'radial-gradient(circle at 15% 0%, rgba(240, 163, 94, 0.12), transparent 45%), radial-gradient(circle at 90% 100%, rgba(138, 180, 248, 0.10), transparent 40%)',
        py: { xs: 4, md: 8 },
      }}
    >
      <Container maxWidth="lg">
        <Stack spacing={4}>
          <Stack spacing={1.5} sx={{ alignItems: 'flex-start' }}>
            <Chip
              icon={<GavelOutlinedIcon />}
              label="Social-Media Hate-Speech Detection"
              color="primary"
              variant="outlined"
            />
            <Typography variant="h1">
              Paste a post or comment and get a moderation verdict instantly.
            </Typography>
            <Typography
              variant="body1"
              color="text.secondary"
              sx={{ maxWidth: 560 }}
            >
              A deployment-ready moderation interface for classifying text as
              hate, offensive, or normal. Every result identifies the backend
              that produced it, including the built-in placeholder demo.
            </Typography>
            {backendStatus === 'placeholder' ? (
              <Alert severity="info" sx={{ maxWidth: 680 }}>
                Demo mode is active. Predictions use a deterministic keyword
                heuristic until a trained model artifact is attached.
              </Alert>
            ) : null}
          </Stack>

          <Box
            sx={{
              display: 'grid',
              gap: 3,
              gridTemplateColumns: { xs: '1fr', md: '1.1fr 0.9fr' },
              alignItems: 'stretch',
            }}
          >
            <Card>
              <CardContent
                component="form"
                onSubmit={handleSubmit}
                sx={{ display: 'flex', flexDirection: 'column', gap: 2.5, p: 3 }}
              >
                <TextField
                  multiline
                  minRows={6}
                  maxRows={12}
                  fullWidth
                  label="Text to analyze"
                  placeholder="Paste a social-media post or comment here…"
                  value={text}
                  onChange={(event) =>
                    setText(event.target.value.slice(0, MAX_TEXT_CHARS))
                  }
                  helperText={`${text.length}/${MAX_TEXT_CHARS} characters`}
                />

                <Stack
                  direction={{ xs: 'column', sm: 'row' }}
                  spacing={2}
                  sx={{ alignItems: { xs: 'stretch', sm: 'center' } }}
                >
                  <Button
                    type="submit"
                    variant="contained"
                    size="large"
                    disabled={!text.trim() || loading}
                    startIcon={
                      loading ? (
                        <CircularProgress size={18} color="inherit" />
                      ) : (
                        <ForumOutlinedIcon />
                      )
                    }
                  >
                    {loading ? 'Analyzing…' : 'Analyze text'}
                  </Button>
                  <Chip
                    label={`Backend: ${prediction?.model_backend ?? backendStatus ?? 'checking…'}`}
                    size="small"
                    variant="outlined"
                    sx={{ maxWidth: '100%' }}
                  />
                </Stack>

                <Stack
                  direction="row"
                  spacing={1}
                  sx={{ flexWrap: 'wrap', gap: 1 }}
                >
                  {SAMPLE_TEXTS.map((sample) => (
                    <Chip
                      key={sample}
                      label={`${sample.slice(0, 42)}…`}
                      size="small"
                      onClick={() => setText(sample)}
                    />
                  ))}
                </Stack>

                {error ? <Alert severity="error">{error}</Alert> : null}
                {prediction && meta ? (
                  <Alert
                    severity={meta.severity}
                    icon={
                      prediction.label === 'hate' ? (
                        <ReportGmailerrorredOutlinedIcon />
                      ) : prediction.label === 'offensive' ? (
                        <WarningAmberOutlinedIcon />
                      ) : (
                        <SentimentSatisfiedAltOutlinedIcon />
                      )
                    }
                  >
                    <strong>{meta.title.toUpperCase()}</strong> with{' '}
                    {formatPercent(prediction.confidence)} confidence.{' '}
                    {meta.blurb}
                  </Alert>
                ) : null}
              </CardContent>
            </Card>

            <Card>
              <CardContent sx={{ p: 3 }}>
                <Typography variant="overline" color="text.secondary">
                  Class probabilities
                </Typography>
                <Stack spacing={2} sx={{ mt: 2 }}>
                  <ProbabilityRow
                    label="Hate"
                    value={prediction ? prediction.probabilities.hate : null}
                    color="error"
                  />
                  <ProbabilityRow
                    label="Offensive"
                    value={
                      prediction ? prediction.probabilities.offensive : null
                    }
                    color="warning"
                  />
                  <ProbabilityRow
                    label="Normal"
                    value={prediction ? prediction.probabilities.normal : null}
                    color="success"
                  />
                </Stack>

                <Divider sx={{ my: 2.5 }} />

                <Stack spacing={1.5} divider={<Divider />}>
                  {[
                    [
                      'Model backend',
                      prediction ? prediction.model_backend : '--',
                    ],
                    ['Labels', 'hate / offensive / normal'],
                    ['API route', '/predict'],
                  ].map(([label, value]) => (
                    <Stack
                      key={label}
                      direction="row"
                      sx={{ justifyContent: 'space-between' }}
                    >
                      <Typography variant="body2" color="text.secondary">
                        {label}
                      </Typography>
                      <Typography variant="body2" sx={{ fontWeight: 600 }}>
                        {value}
                      </Typography>
                    </Stack>
                  ))}
                </Stack>

                {prediction?.model_backend === 'placeholder' ? (
                  <Alert severity="info" sx={{ mt: 2.5 }}>
                    Running in placeholder mode — a keyword heuristic stands in
                    until the trained model is attached to the backend.
                  </Alert>
                ) : null}
              </CardContent>
            </Card>
          </Box>

          <Box
            sx={{
              display: 'grid',
              gap: 3,
              gridTemplateColumns: { xs: '1fr', md: 'repeat(3, 1fr)' },
            }}
          >
            {(
              [
                [
                  'Datasets',
                  'HateXplain · Davidson · Jigsaw',
                  'Designed for benchmark workflows that combine three established moderation corpora.',
                ],
                [
                  'Models',
                  'TF-IDF baselines vs Transformers',
                  'Supports a serialized scikit-learn probability pipeline or a fine-tuned transformer drop-in.',
                ],
                [
                  'Evaluation',
                  'Macro & weighted F1',
                  'Recommended reporting includes per-class precision, recall, F1, confusion matrices, and error analysis.',
                ],
              ] as const
            ).map(([overline, title, body]) => (
              <Card key={overline}>
                <CardContent sx={{ p: 3 }}>
                  <Typography variant="overline" color="text.secondary">
                    {overline}
                  </Typography>
                  <Typography variant="h2" sx={{ mt: 1, mb: 1 }}>
                    {title}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {body}
                  </Typography>
                </CardContent>
              </Card>
            ))}
          </Box>
        </Stack>
      </Container>
    </Box>
  );
}
