import { create } from 'zustand'
import { persist } from 'zustand/middleware'

type CurrentExperiment = {
  name: string | null
  setName: (name: string | null) => void
}

/** The experiment shown in the sidebar (remembered per browser). */
export const useCurrentExperiment = create<CurrentExperiment>()(
  persist((set) => ({ name: null, setName: (name) => set({ name }) }), {
    name: 'stereodic-current-experiment',
  })
)
