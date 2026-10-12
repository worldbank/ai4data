import clsx from 'clsx';
import Link from '@docusaurus/Link';
import Heading from '@theme/Heading';
import {cookbooks, groups} from '@site/src/content/cookbooks';
import {countWord} from '@site/src/content/visibility';
import styles from './styles.module.css';

export default function CookbookCta() {
  return (
    <section className={styles.section}>
      <div className="container">
        <div className={styles.layout}>
          <div className={styles.text}>
            <span className="eyebrow">For national statistical organizations</span>
            <Heading as="h2" className={styles.title}>
              Practical guides for AI-ready data
            </Heading>
            <p className={styles.lede}>
              {countWord(cookbooks.length, true)} cookbooks, each organized around the questions a team asks
              about one part of its work: dissemination, microdata, documents,
              metadata, agent interfaces, models in production, synthetic data,
              small and open models, evaluation, monitoring of use, and
              datasets for machine learning. Every chapter has recipes at three
              maturity levels, scripts that run on a shared example, open
              standards, and a checklist. Generative AI is optional in the
              first steps, which are complete metadata, open data, stable
              identifiers, and documented interfaces.
            </p>
            <div className={styles.actions}>
              <Link className={clsx('button button--md', styles.button)} to="/cookbook/">
                The cookbooks
              </Link>
              <Link className={styles.secondary} to="/modern-nso">
                A modern statistical office →
              </Link>
            </div>
          </div>
          <div className={styles.groups}>
            {groups.map((g) => (
              <div key={g.title} className={styles.group}>
                <span className={styles.groupTitle}>{g.title}</span>
                <ul className={styles.list}>
                  {g.ids.map((id) => cookbooks.find((c) => c.id === id)).map((c) => (
                    <li key={c.id}>
                      <Link className={styles.item} to={c.to}>
                        {c.title.replace('Practical Guide to ', '')}
                      </Link>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
