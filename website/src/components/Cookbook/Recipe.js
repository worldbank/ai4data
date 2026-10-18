import styles from './styles.module.css';

// A recipe is a task with a defined result. The header shows the maturity
// level it belongs to, the skills needed, and an indicative time.
export default function Recipe({title, level, skills, time, children}) {
  return (
    <section className={styles.recipe}>
      <header className={styles.recipeHead}>
        <span className={styles.recipeKicker}>Recipe</span>
        <h3 className={styles.recipeTitle}>{title}</h3>
        <dl className={styles.recipeMeta}>
          {level && (
            <div>
              <dt>Level</dt>
              <dd>{level}</dd>
            </div>
          )}
          {skills && (
            <div>
              <dt>Skills</dt>
              <dd>{skills}</dd>
            </div>
          )}
          {time && (
            <div>
              <dt>Indicative time</dt>
              <dd>{time}</dd>
            </div>
          )}
        </dl>
      </header>
      <div className={styles.recipeBody}>{children}</div>
    </section>
  );
}
