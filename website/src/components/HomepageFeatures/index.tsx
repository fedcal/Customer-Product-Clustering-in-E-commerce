import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  emoji: string;
  description: ReactNode;
  to: string;
  cta: string;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Teoria, spiegata bene',
    emoji: '📚',
    description: (
      <>
        Cinque articoli che costruiscono progressivamente la conoscenza:
        feature RFM point-in-time, K-Means vs alternative, split temporali
        senza leakage, metriche multiclasse e pipeline riproducibile.
        Con LaTeX e codice eseguibile.
      </>
    ),
    to: '/docs/category/teoria',
    cta: 'Esplora la teoria',
  },
  {
    title: 'Pipeline modulare e riproducibile',
    emoji: '⚙️',
    description: (
      <>
        Codice modulare in Python con scikit-learn e XGBoost,
        <code> StandardScaler </code> separato dal modello di clustering,
        two-snapshot temporale anti-leakage e CLI
        <code> ecom-cluster --quick</code> per smoke test.
      </>
    ),
    to: '/docs/scelte-tecniche/architettura',
    cta: 'Vedi architettura',
  },
  {
    title: 'Trade-off documentati',
    emoji: '🎯',
    description: (
      <>
        Ogni scelta è motivata: perché K-Means come modello primario,
        perché GMM come confronto, perché RandomForest e XGBoost per il
        cluster futuro, e come interpretare la confusion matrix sui
        cluster di alto valore.
      </>
    ),
    to: '/docs/scelte-tecniche/scelte-modello',
    cta: 'Leggi le decisioni',
  },
];

function Feature({title, emoji, description, to, cta}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className={styles.featureCard}>
        <div className={styles.featureEmoji} role="img" aria-label={title}>
          {emoji}
        </div>
        <Heading as="h3" className={styles.featureTitle}>
          {title}
        </Heading>
        <p className={styles.featureDesc}>{description}</p>
        <Link className={clsx('button button--primary button--sm', styles.featureCta)} to={to}>
          {cta}
        </Link>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="text--center" style={{marginBottom: '3rem'}}>
          <Heading as="h2">Cosa trovi in questa documentazione</Heading>
          <p style={{color: 'var(--ifm-color-emphasis-700)', maxWidth: 700, margin: '0 auto'}}>
            Ogni sezione è autocontenuta. Leggi nell&apos;ordine se vuoi una progressione
            didattica, oppure salta direttamente all&apos;argomento che ti serve.
          </p>
        </div>
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
